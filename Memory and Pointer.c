#include <stdio.h>
#include <stdlib.h>

/*
    Demonstration: Safe Memory and Pointer Handling in C (C89 Standard)
    Features:
        - Stack variables and pointers
        - Dynamic memory allocation and checks
        - Safe pointer usage (initialization, after-free, boundary)
        - Best practices for beginners
*/

int main() {
    /* 1. Stack variable and pointer */
    int num = 42;
    int *ptr = NULL; /* Always initialize pointer to NULL to avoid wild pointer */
    ptr = &num;

    printf("Stack variable address: %p, value: %d\n", (void*)&num, num);
    printf("Pointer ptr points to address: %p, value: %d\n", (void*)ptr, *ptr);

    /* 2. Dynamic memory allocation: always check malloc! */
    int *dyn_int = NULL;
    dyn_int = (int*) malloc(sizeof(int)); /* Allocate memory for one integer */
    if (dyn_int == NULL) {
        printf("Memory allocation failed!\n");
        return 1; /* Stop program if allocation failed */
    }
    *dyn_int = 99;
    printf("Dynamically allocated int address: %p, value: %d\n", (void*)dyn_int, *dyn_int);

    /* Always free allocated memory and set pointer to NULL */
    free(dyn_int);
    dyn_int = NULL;

    /* 3. Static array and safe access */
    int arr[3] = {7, 8, 9};
    int i = 0;
    printf("Static array contents: ");
    for (i = 0; i < 3; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    /* 4. Allocate memory to store several integers */
    int n = 5;
    int *array_ptr = NULL;
    array_ptr = (int*) malloc(n * sizeof(int));
    if (array_ptr == NULL) {
        printf("Dynamic array allocation failed!\n");
        return 1;
    }
    /* Store values */
    for (i = 0; i < n; ++i) {
        array_ptr[i] = (i + 1) * 10;
    }
    printf("Dynamic array contents: ");
    for (i = 0; i < n; ++i) {
        printf("%d ", array_ptr[i]);
    }
    printf("\n");

    /* Release memory and reset pointer */
    free(array_ptr);
    array_ptr = NULL;

    /* 5. Safe use of NULL pointers */
    int *safety = NULL;
    if (safety == NULL) {
        printf("Pointer 'safety' is NULL. Safe: do not dereference!\n");
    }

    /* 6. Never use pointers after free! (no demonstration, just a reminder) */
    printf("\nSafety summary:\n");
    printf("- Always initialize pointers when declaring.\n");
    printf("- After malloc/calloc, always check if memory was allocated successfully.\n");
    printf("- Always free memory after you're done using it.\n");
    printf("- After free, set pointer to NULL to avoid using a dangling pointer.\n");
    printf("- Never access memory out of array bounds.\n");
    printf("- Never dereference NULL or uninitialized pointers.\n");

    return 0;
}
