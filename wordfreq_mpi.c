#include <ctype.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_WORD_LEN 100
#define HASH_TABLE_SIZE 16384     // Larger for better distribution
#define MAX_BUFFER_SIZE (1 << 26) // 64MB max buffer
#define CHUNK_SIZE 8192           // File read chunk size

int verbose = 1;
#define LOG(rank, fmt, ...)                                                    \
  do {                                                                         \
    if (verbose)                                                               \
      fprintf(stderr, "[Rank %d] " fmt "\n", rank, ##__VA_ARGS__);             \
  } while (0)

typedef struct WordNode {
  char word[MAX_WORD_LEN];
  int count;
  struct WordNode *next;
} WordNode;

typedef struct {
  WordNode **buckets;
  int size;
  int items;
} HashMap;

HashMap *create_hashmap(int size);
void free_hashmap(HashMap *map);
void insert_word(HashMap *map, const char *word);
HashMap *process_file(const char *filename, const char *delims, int rank);
void serialize_hashmap(HashMap *map, char **buffer, int *length, int rank);
void deserialize_hashmap(HashMap *map, const char *buffer, int length,
                         int rank);

// FNV-1a hash function
unsigned int hash(const char *word, int size) {
  unsigned int h = 2166136261u;
  while (*word) {
    h ^= (unsigned char)(tolower(*word++));
    h *= 16777619u;
  }
  return h % size;
}

HashMap *create_hashmap(int size) {
  HashMap *map = malloc(sizeof(HashMap));
  if (!map) {
    LOG(0, "Failed to allocate hashmap");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  map->buckets = calloc(size, sizeof(WordNode *));
  if (!map->buckets) {
    free(map);
    LOG(0, "Failed to allocate hashmap buckets");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  map->size = size;
  map->items = 0;
  return map;
}

void free_hashmap(HashMap *map) {
  if (!map)
    return;
  free(map->buckets);
  free(map);
}

void insert_word(HashMap *map, const char *word) {
  unsigned int h = hash(word, map->size);
  WordNode *node = map->buckets[h];

  while (node) {
    if (strncasecmp(node->word, word, MAX_WORD_LEN) == 0) {
      node->count++;
      return;
    }
    node = node->next;
  }

  node = malloc(sizeof(WordNode));

  strncpy(node->word, word, MAX_WORD_LEN - 1);
  node->word[MAX_WORD_LEN - 1] = '\0';
  node->count = 1;
  node->next = map->buckets[h];
  map->buckets[h] = node;
  map->items++;
}

int is_delimiter(char c, const char *delims) {
  while (*delims) {
    if (c == *delims++)
      return 1;
  }
  return 0;
}

HashMap *process_file(const char *filename, const char *delims, int rank) {
  LOG(rank, "Opening file %s", filename);
  FILE *file = fopen(filename, "r");
  if (!file) {
    LOG(rank, "Failed to open file %s", filename);
    return NULL;
  }

  HashMap *map = create_hashmap(HASH_TABLE_SIZE);
  char *buffer = malloc(CHUNK_SIZE);
  char word[MAX_WORD_LEN];
  int word_len = 0;

  if (!buffer) {
    LOG(rank, "Failed to allocate file buffer");
    fclose(file);
    free_hashmap(map);
    return NULL;
  }

  size_t bytes;
  while ((bytes = fread(buffer, 1, CHUNK_SIZE - 1, file)) > 0) {
    buffer[bytes] = '\0';
    for (size_t i = 0; i < bytes; i++) {
      char c = buffer[i];
      if (is_delimiter(c, delims) || c == '\n' || c == '\r') {
        if (word_len > 0) {
          word[word_len] = '\0';
          insert_word(map, word);
          word_len = 0;
        }
      } else if (word_len < MAX_WORD_LEN - 1) {
        word[word_len++] = c;
      }
    }
  }

  if (word_len > 0) {
    word[word_len] = '\0';
    insert_word(map, word);
  }

  if (ferror(file)) {
    LOG(rank, "Error reading file %s", filename);
    free(buffer);
    fclose(file);
    free_hashmap(map);
    return NULL;
  }

  free(buffer);
  fclose(file);
  LOG(rank, "Processed file %s, items: %d", filename, map->items);
  return map;
}

void merge_hashmaps(HashMap *dest, HashMap *src) {
  if (!src)
    return;
  for (int i = 0; i < src->size; i++) {
    WordNode *node = src->buckets[i];
    while (node) {
      for (int j = 0; j < node->count; j++)
        insert_word(dest, node->word);
      node = node->next;
    }
  }
}

void serialize_hashmap(HashMap *map, char **buffer, int *length, int rank) {
  LOG(rank, "Starting serialization, items: %d", map->items);
  *length = map->items * (MAX_WORD_LEN + 12);
  if (*length > MAX_BUFFER_SIZE) {
    LOG(rank, "Buffer size %d exceeds max %d", *length, MAX_BUFFER_SIZE);
    free_hashmap(map);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  *buffer = malloc(*length);
  if (!*buffer) {
    LOG(rank, "Failed to allocate serialization buffer");
    free_hashmap(map);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  char *ptr = *buffer;
  int written = 0;
  for (int i = 0; i < map->size; i++) {
    WordNode *node = map->buckets[i];
    while (node) {
      int len =
          snprintf(ptr, *length - written, "%s:%d\n", node->word, node->count);
      if (len < 0 || written + len >= *length) {
        LOG(rank, "Buffer overflow during serialization, written: %d, len: %d",
            written, len);
        free(*buffer);
        free_hashmap(map);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      ptr += len;
      written += len;
      node = node->next;
    }
  }

  *length = written ? written : 1;
  if (!written)
    (*buffer)[0] = '\0';
  LOG(rank, "Serialized %d bytes", *length);
}

void deserialize_hashmap(HashMap *map, const char *buffer, int length,
                         int rank) {
  LOG(rank, "Starting deserialization, length: %d", length);
  if (length <= 1 && buffer[0] == '\0')
    return;

  char *copy = strndup(buffer, length);
  if (!copy) {
    LOG(rank, "Failed to allocate deserialization buffer");
    free_hashmap(map);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  char *line = strtok(copy, "\n");
  while (line) {
    char *colon = strchr(line, ':');
    if (colon) {
      *colon = '\0';
      int count = atoi(colon + 1);
      if (count > 0) {
        for (int i = 0; i < count; i++)
          insert_word(map, line);
      }
    }
    line = strtok(NULL, "\n");
  }
  free(copy);
}

int compare_words(const void *a, const void *b) {
  WordNode *wa = (WordNode *)a;
  WordNode *wb = (WordNode *)b;

  if (wb->count != wa->count)
    return wb->count - wa->count;

  return strncasecmp(wa->word, wb->word, MAX_WORD_LEN);
}

void print_results(HashMap *map, int top_n) {
  WordNode *words = malloc(map->items * sizeof(WordNode));
  int idx = 0;

  for (int i = 0; i < map->size; i++) {
    WordNode *current = map->buckets[i];
    while (current) {
      strncpy(words[idx].word, current->word, MAX_WORD_LEN);
      words[idx].word[MAX_WORD_LEN - 1] = '\0';
      words[idx].count = current->count;
      idx++;
      current = current->next;
    }
  }

  qsort(words, map->items, sizeof(WordNode), compare_words);

  printf("\nTop %d words by frequency:\n", top_n);
  printf("----------------------------\n");
  printf("| %-16s | %-7s |\n", "Word", "Count");
  printf("----------------------------\n");

  for (int i = 0; i < map->items && i < top_n; i++) {
    printf("| %-16s | %-7d |\n", words[i].word, words[i].count);
  }
  printf("----------------------------\n");

  free(words);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  char *delims = " ,.!?;:\n";
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 2) {
    if (rank == 0)
      fprintf(stderr, "Usage: %s <file1> [file2 ...]\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  double start_time = MPI_Wtime();
  int num_files = argc - 1;
  LOG(rank, "Processing %d files", num_files);

  // Process files
  HashMap *local_map = create_hashmap(HASH_TABLE_SIZE);
  for (int i = rank; i < argc - 1; i += size) {
    LOG(rank, "Assigned file: %s", argv[i + 1]);
    const char *filename = argv[i + 1];
    HashMap *tmp = process_file(filename, delims, rank);
    if (tmp) {
      merge_hashmaps(local_map, tmp);
      free_hashmap(tmp);
    }
  }

  // Serialize local map
  char *send_buffer;
  int send_length;
  serialize_hashmap(local_map, &send_buffer, &send_length, rank);

  // Gather lengths
  int *recv_lengths = NULL;
  int *displs = NULL;
  char *recv_buffer = NULL;
  if (rank == 0) {
    recv_lengths = malloc(size * sizeof(int));
    displs = malloc(size * sizeof(int));
    if (!recv_lengths || !displs) {
      LOG(0, "Failed to allocate gather buffers");
      free_hashmap(local_map);
      free(send_buffer);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  MPI_Gather(&send_length, 1, MPI_INT, recv_lengths, 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  // Gather data
  if (rank == 0) {
    int total_length = 0;
    for (int i = 0; i < size; i++) {
      displs[i] = total_length;
      total_length += recv_lengths[i];
    }
    if (total_length > MAX_BUFFER_SIZE) {
      LOG(0, "Total gathered size %d exceeds max %d", total_length,
          MAX_BUFFER_SIZE);
      free(recv_lengths);
      free(displs);
      free_hashmap(local_map);
      free(send_buffer);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    recv_buffer = malloc(total_length);
    if (!recv_buffer) {
      LOG(0, "Failed to allocate receive buffer");
      free(recv_lengths);
      free(displs);
      free_hashmap(local_map);
      free(send_buffer);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  MPI_Gatherv(send_buffer, send_length, MPI_CHAR, recv_buffer, recv_lengths,
              displs, MPI_CHAR, 0, MPI_COMM_WORLD);
  free(send_buffer);

  // Process gathered data
  if (rank == 0) {
    HashMap *global_map = create_hashmap(HASH_TABLE_SIZE);
    merge_hashmaps(global_map, local_map);
    for (int i = 1; i < size; i++) {
      if (recv_lengths[i] > 0) {
        deserialize_hashmap(global_map, recv_buffer + displs[i],
                            recv_lengths[i], rank);
      }
    }
    double end_time = MPI_Wtime();
    LOG(rank, "Processing time: %f seconds", end_time - start_time);
    print_results(global_map, 10);
    free_hashmap(global_map);
    free(recv_buffer);
    free(recv_lengths);
    free(displs);
  }

  free_hashmap(local_map);
  MPI_Finalize();
  return 0;
}
