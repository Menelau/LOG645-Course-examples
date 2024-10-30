/*
Exemple of thread programming using C++ Boost

Compilation line
 g++ -o multithread multithread.cpp -lboost_system -lboost_thread -lpthread
*/

#include <iostream>
#include <boost/thread/thread.hpp>
// the number of child threads to create
#define NB_ENFANTS 8

// Function that each thread will execute
void enfant(int k) {
    std::cout << "I am child " << k << std::endl;
}

int main() {
    // Create a thread group to manage multiple threads
    boost::thread_group mesthreads;

    // Create threads
    for (auto i = 0; i < NB_ENFANTS; i++) {
        // Use boost::bind to bind the 'enfant' function with argument 'i'
        // This creates a callable object that will call 'enfant(i)'
        boost::thread *th = new boost::thread(boost::bind(&enfant, i));
        
        // Add the created thread to the thread group 'mesthreads'
        mesthreads.add_thread(th);
    }

    // Wait for all threads in 'mesthreads' to finish execution
    mesthreads.join_all();

    return EXIT_SUCCESS;
}
