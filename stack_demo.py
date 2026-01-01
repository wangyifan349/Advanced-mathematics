#include <stdio.h>

#define MAXSIZE 10 /* 栈最大容量，这是演示用，可以调大 */
typedef struct {
    int data[MAXSIZE]; /* 存数据的数组 */
    int top;           /* 指向栈顶元素下标，空栈时为-1 */
} Stack;

/* -- 初始化栈，top归-1 -- */
void init_stack(Stack *s) {
    s->top = -1;
}

/* -- 判断栈是否为空 -- */
int is_empty(Stack *s) {
    return (s->top == -1);
}

/* -- 判断栈是否已满 -- */
int is_full(Stack *s) {
    return (s->top == MAXSIZE - 1);
}

/* -- 入栈，成功返回1，失败返回0 -- */
int push(Stack *s, int val) {
    if (is_full(s)) {
        return 0; /* 已满，不能入栈 */
    }
    s->top = s->top + 1;
    s->data[s->top] = val;
    return 1;
}

/* -- 出栈，*val获得弹出的值，成功返回1，失败返回0 -- */
int pop(Stack *s, int *val) {
    if (is_empty(s)) {
        return 0; /* 空栈不能出栈 */
    }
    *val = s->data[s->top];
    s->top = s->top - 1;
    return 1;
}

/* -- 读取栈顶元素，*val获得值，成功返回1，失败返回0 -- */
int peek(Stack *s, int *val) {
    if (is_empty(s)) {
        return 0;
    }
    *val = s->data[s->top];
    return 1;
}

/* -- 打印栈内容，仅作演示 -- */
void print_stack(Stack *s) {
    int i;
    printf("栈底 --> ");
    if (is_empty(s)) {
        printf("[空栈]");
    } else {
        for(i = 0; i <= s->top; ++i) {
            printf("%d ", s->data[i]);
        }
    }
    printf("<-- 栈顶\n");
}

/* -- 主程序演示 -- */
int main() {
    Stack s;
    int val;
    int i;

    init_stack(&s);
    printf("初始化后：\n");
    print_stack(&s);

    printf("\n依次入栈 10个元素：\n");
    for(i = 1; i <= MAXSIZE; ++i) {
        if(push(&s, i * 10)) {
            printf("成功入栈: %d\n", i * 10);
        } else {
            printf("入栈失败: 栈已满\n");
        }
        print_stack(&s);
    }
    /* 尝试再入栈，应该失败 */
    printf("再入栈1000（模拟栈满）：\n");
    if(!push(&s, 1000)) {
        printf("提示：栈已满，无法入栈1000\n");
    }
    print_stack(&s);

    /* 取栈顶，不弹出 */
    printf("\n读取当前栈顶：\n");
    if(peek(&s, &val)) {
        printf("栈顶元素为: %d\n", val);
    }

    /* 依次出栈 */
    printf("\n依次出栈到空：\n");
    while(pop(&s, &val)) {
        printf("弹出: %d 剩余: ", val);
        print_stack(&s);
    }

    /* 空栈出栈，peek */
    printf("\n测试空栈操作：\n");
    if(!pop(&s, &val)) {
        printf("提示：空栈无法出栈\n");
    }
    if(!peek(&s, &val)) {
        printf("提示：空栈无法读取栈顶\n");
    }

    return 0;
}
