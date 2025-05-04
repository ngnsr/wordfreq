#include <ctype.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORD_LEN 100
#define HASH_TABLE_SIZE 16384

int verbose = 0;
#define LOG(...)                                                               \
  do {                                                                         \
    if (verbose)                                                               \
      printf(__VA_ARGS__);                                                     \
  } while (0)

typedef struct WordNode {
  char *word;
  int count;
  int hash;
  struct WordNode *next;
} WordNode;

typedef struct {
  char *word;
  int count;
} WordFreq;

typedef struct {
  WordNode **buckets;
  int size;
  int items;
} HashMap;

HashMap *create_hashmap(int size) {
  HashMap *map = malloc(sizeof(HashMap));
  map->size = size;
  map->items = 0;
  map->buckets = calloc(size, sizeof(WordNode *));
  return map;
}

unsigned int hash(const char *word, int size) {
  unsigned int h = 2166136261u;
  while (*word) {
    h ^= (unsigned char)(tolower(*word++));
    h *= 16777619u;
  }
  return h % size;
}

void insert_word(HashMap *map, const char *word) {
  unsigned int h = hash(word, map->size);
  WordNode *current = map->buckets[h];

  while (current) {
    if (strncasecmp(current->word, word, MAX_WORD_LEN) == 0) {
      current->count++;
      return;
    }
    current = current->next;
  }

  WordNode *node = malloc(sizeof(WordNode));
  if (!node) {
    fprintf(stderr, "Memory allocation error\n");
    exit(1);
  }

  node->word = strdup(word);
  node->count = 1;
  node->hash = h;
  node->next = map->buckets[h];
  map->buckets[h] = node;
  map->items++;
}

void merge_hashmaps(HashMap *dest, HashMap *src) {
#pragma omp critical
  for (int i = 0; i < src->size; i++) {
    WordNode *current = src->buckets[i];
    while (current) {
      unsigned int h = hash(current->word, dest->size);
      WordNode *dest_node = dest->buckets[h];
      int found = 0;

      while (dest_node && !found) {
        if (strcmp(dest_node->word, current->word) == 0) {
          dest_node->count += current->count;
          found = 1;
        }
        dest_node = dest_node->next;
      }

      if (!found) {
        WordNode *new_node = malloc(sizeof(WordNode));
        if (!new_node || !(new_node->word = strdup(current->word))) {
          fprintf(stderr, "Memory allocation error\n");
          exit(1);
        }
        new_node->count = current->count;
        new_node->hash = current->hash;
        new_node->next = dest->buckets[h];
        dest->buckets[h] = new_node;
        dest->items++;
      }

      current = current->next;
    }
  }
}

void free_hashmap(HashMap *map) {
  for (int i = 0; i < map->size; i++) {
    WordNode *current = map->buckets[i];
    while (current) {
      WordNode *temp = current;
      current = current->next;
      free(temp->word);
      free(temp);
    }
  }
  free(map->buckets);
  free(map);
}

int is_delimiter(char c, const char *delimiters) {
  while (*delimiters) {
    if (c == *delimiters)
      return 1;
    delimiters++;
  }
  return 0;
}

HashMap *process_file_sync(const char *filename, const char *delimiters) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Error opening file %s\n", filename);
    return NULL;
  }

  HashMap *word_map = create_hashmap(HASH_TABLE_SIZE);
  char word[MAX_WORD_LEN];
  int word_len = 0;
  int c;

  while ((c = fgetc(file)) != EOF) {
    if (is_delimiter(c, delimiters) || c == '\n' || c == '\r') {
      if (word_len > 0) {
        word[word_len] = '\0';
        insert_word(word_map, word);
        word_len = 0;
      }
    } else if (word_len < MAX_WORD_LEN - 1) {
      word[word_len++] = c;
    }
  }

  if (word_len > 0) {
    word[word_len] = '\0';
    insert_word(word_map, word);
  }

  fclose(file);
  LOG("Processed file %s, items: %d", filename, word_map->items);
  return word_map;
}

HashMap *process_files_parallel(char **filenames, int num_files,
                                const char *delimiters, int num_threads) {
  HashMap *global_map = create_hashmap(HASH_TABLE_SIZE);

  LOG("Starting parallel processing with %d threads...\n", num_threads);
  omp_set_num_threads(num_threads);

#pragma omp parallel shared(global_map, filenames, num_files, delimiters)
  {
    int thread_id = omp_get_thread_num();
    HashMap *local_map = create_hashmap(HASH_TABLE_SIZE);

    if (!local_map) {
      fprintf(stderr, "Error allocating hash table for thread %d\n", thread_id);
      exit(1);
    }

    LOG("Thread %d started\n", thread_id);
#pragma omp for schedule(dynamic)
    for (int i = 0; i < num_files; i++) {
      LOG("Thread %d processing file %s\n", thread_id, filenames[i]);
      HashMap *file_map = process_file_sync(filenames[i], delimiters);
      if (file_map) {
        merge_hashmaps(local_map, file_map);
        free_hashmap(file_map);
      }
    }
    LOG("Thread %d finished processing\n", thread_id);
    LOG("Thread %d merging results...\n", thread_id);
    merge_hashmaps(global_map, local_map);
    LOG("Thread %d merge complete\n", thread_id);

    free_hashmap(local_map);
  }

  return global_map;
}

HashMap *process_files_sync(char **filenames, int num_files,
                            const char *delimiters) {
  HashMap *global_map = create_hashmap(HASH_TABLE_SIZE);
  for (int i = 0; i < num_files; i++) {
    HashMap *file_map = process_file_sync(filenames[i], delimiters);
    if (file_map) {
      merge_hashmaps(global_map, file_map);
      free_hashmap(file_map);
    }
  }
  return global_map;
}

int compare_words(const void *a, const void *b) {
  WordFreq *wa = (WordFreq *)a;
  WordFreq *wb = (WordFreq *)b;

  if (wb->count != wa->count)
    return wb->count - wa->count;

  return strcmp(wa->word, wb->word);
}

void print_results(HashMap *map, int top_n) {
  WordFreq *words = malloc(map->items * sizeof(WordFreq));
  int idx = 0;

  for (int i = 0; i < map->size; i++) {
    WordNode *current = map->buckets[i];
    while (current) {
      words[idx].word = current->word;
      words[idx].count = current->count;
      idx++;
      current = current->next;
    }
  }

  qsort(words, map->items, sizeof(WordFreq), compare_words);

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

void run_benchmark(char **filenames, int num_files, const char *delimiters) {
  printf("\nBenchmark results:\n");
  printf("--------------------------------------------------\n");
  printf("| %-12s | %-15s | %-15s |\n", "Method", "Time (s)", "Speedup");
  printf("--------------------------------------------------\n");

  double sync_time;
  {
    LOG("Running sync version...\n");
    double start = omp_get_wtime();
    HashMap *sync_map = process_files_sync(filenames, num_files, delimiters);
    double end = omp_get_wtime();

    sync_time = end - start;
    printf("| %-12s | %-15.6f | %-15.6f |\n", "Sync", sync_time, 1.0);
    LOG("Unique words in sync: %d\n", sync_map->items);
    free_hashmap(sync_map);
  }

  int thread_counts[] = {2, 4, 8};

  for (int i = 0; i < sizeof(thread_counts) / sizeof(thread_counts[0]); i++) {
    int threads = thread_counts[i];

    LOG("Running parallel version with %d threads...\n", threads);
    double start = omp_get_wtime();
    HashMap *parallel_map =
        process_files_parallel(filenames, num_files, delimiters, threads);
    double end = omp_get_wtime();

    double parallel_time = end - start;
    double speedup = sync_time / parallel_time;

    printf("| %-5s (%d) | %-15.6f | %-15.6f |\n", "Parallel", threads,
           parallel_time, speedup);
    free_hashmap(parallel_map);
  }

  printf("--------------------------------------------------\n");
}

void print_usage() {
  printf("Usage: program [options] file1 [file2 ...]\n");
  printf("Options:\n");
  printf("  -n <num>          Number of threads (default: 4)\n");
  printf("  -d <delimiters>   Delimiters (default: \" ,.!?;:\")\n");
  printf("  -t <num>          Top N words to print (default: 10)\n");
  printf("  -b                Run benchmark mode\n");
  printf("  -r                Show top N words\n");
  printf("  -v                Disable verbose output\n");
  printf("  -h                Show help\n");
}

int main(int argc, char **argv) {
  char *delimiters = " ,.!?;:";
  int top_n = 10;
  int run_bench = 0;
  int print_list = 0;
  int num_threads = 4;

  int i;
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-')
      break;

    switch (argv[i][1]) {
    case 'd':
      if (i + 1 < argc)
        delimiters = argv[++i];
      break;
    case 't':
      if (i + 1 < argc)
        top_n = atoi(argv[++i]);
      break;
    case 'b':
      run_bench = 1;
      break;
    case 'r':
      print_list = 1;
      break;
    case 'n':
      num_threads = atoi(argv[++i]);
      if (num_threads <= 0)
        return 1;
      break;
    case 'v':
      verbose = 1;
      break;
    case 'h':
      print_usage();
      return 0;
    default:
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      print_usage();
      return 1;
    }
  }

  if (i >= argc) {
    fprintf(stderr, "Error: No input files provided\n");
    print_usage();
    return 1;
  }

  int num_files = argc - i;
  char **filenames = &argv[i];

  LOG("Starting word frequency count on %d file(s)\n", num_files);
  LOG("Using delimiters: '%s'\n", delimiters);

  if (run_bench) {
    run_benchmark(filenames, num_files, delimiters);
  } else {
    double start = omp_get_wtime();
    HashMap *map =
        process_files_parallel(filenames, num_files, delimiters, num_threads);
    double end = omp_get_wtime();

    printf("\nExecution time: %.6f seconds\n", end - start);
    if (print_list) {
      print_results(map, top_n);
    }

    free_hashmap(map);
  }

  return 0;
}
