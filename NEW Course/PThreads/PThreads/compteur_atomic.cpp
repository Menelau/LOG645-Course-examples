#include <iostream>
#include <atomic>
#include <pthread.h>

#define NB_ENFANTS 8  // Define the number of child threads

// Atomic counter, providing thread-safe operations without a mutex
static std::atomic<unsigned int> cnt(0);

// Function executed by each thread
void* enfant(void* arg) {
    unsigned int* k = (unsigned int*) arg;
    int ex = 0;
    std::cout << "I am child " << *k << std::endl;

    // Atomic increment (no mutex needed)
    cnt++;

    pthread_exit((void*)&ex);
}

int main() {
    pthread_t t[NB_ENFANTS];  // Array of thread handles
    unsigned int indices[NB_ENFANTS];  // Array to hold indices for each thread
    int* ret;

    // Create threads
    for (unsigned int i = 0; i < NB_ENFANTS; i++) {
        indices[i] = i;  // Assign each thread a unique index
        pthread_create(&t[i], NULL, enfant, (void*)&indices[i]);
    }

    // Join threads
    for (unsigned int i = 0; i < NB_ENFANTS; i++) {
        pthread_join(t[i], (void**)&ret);
    }

    // Output the counter value
    std::cout << "Counter value: " << cnt.load() << std::endl;

    return EXIT_SUCCESS;
}
