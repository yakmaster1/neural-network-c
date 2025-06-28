#include <windows.h>
#include <stdio.h>
#include "draw_window.h"

#define PIXEL_SIZE 10
#define GRID_SIZE 28
#define WINDOW_SIZE (PIXEL_SIZE * GRID_SIZE)

static float input_data[GRID_SIZE * GRID_SIZE] = {0};
static NeuralNetwork *active_network = NULL;

// GENERIERT VON CHATGPT

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

void DrawPixel(HWND hwnd, int cx, int cy) {
    HDC hdc = GetDC(hwnd);
    const float kernel[3][3] = {
        {0.01f, 0.03f, 0.01f},
        {0.03f, 1.00f, 0.03f},
        {0.01f, 0.03f, 0.01f}
    };

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int x = cx + dx;
            int y = cy + dy;

            if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
                int index = y * GRID_SIZE + x;
                float add = kernel[dy + 1][dx + 1] * 255.0f;

                input_data[index] += add;
                if (input_data[index] > 255.0f) input_data[index] = 255.0f;

                int gray = (int)input_data[index];
                if (gray > 255) gray = 255;
                if (gray < 0) gray = 0;

                RECT rect = {
                    x * PIXEL_SIZE,
                    y * PIXEL_SIZE,
                    (x + 1) * PIXEL_SIZE,
                    (y + 1) * PIXEL_SIZE
                };
                HBRUSH brush = CreateSolidBrush(RGB(gray, gray, gray));
                FillRect(hdc, &rect, brush);
                DeleteObject(brush);
            }
        }
    }

    ReleaseDC(hwnd, hdc);
}

void ClearGrid(HWND hwnd) {
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        input_data[i] = 0.0f;
    }
    InvalidateRect(hwnd, NULL, TRUE);
}

void PredictDigit(HWND hwnd) {
    Vector *input = create_v(GRID_SIZE * GRID_SIZE, input_data, INIT);
    set_network_input(active_network, input);
    compute_activation(active_network);
    print_draw_output(active_network);
    dispose_v(input);
}

void start_draw_window(NeuralNetwork *network) {
    active_network = network;
    const char CLASS_NAME[] = "DrawWindowClass";

    WNDCLASS wc = {0};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = CLASS_NAME;
    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        0, CLASS_NAME, "Draw a Digit (ENTER = Predict, C = Clear)",
        WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_SIZE + 16, WINDOW_SIZE + 39,
        NULL, NULL, GetModuleHandle(NULL), NULL
    );

    if (!hwnd) return;

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg = {0};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static int isDrawing = 0;

    switch (uMsg) {
    case WM_LBUTTONDOWN:
        isDrawing = 1;
    case WM_MOUSEMOVE:
        if (isDrawing && (wParam & MK_LBUTTON)) {
            int x = LOWORD(lParam) / PIXEL_SIZE;
            int y = HIWORD(lParam) / PIXEL_SIZE;
            if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE)
                DrawPixel(hwnd, x, y);
        }
        break;
    case WM_LBUTTONUP:
        isDrawing = 0;
        break;
    case WM_KEYDOWN:
        if (wParam == VK_RETURN) {
            PredictDigit(hwnd);
            ClearGrid(hwnd);
        }
        if (wParam == 'C') {
            ClearGrid(hwnd);
        }
        break;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
         for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                 RECT r = {
                     x * PIXEL_SIZE,
                    y * PIXEL_SIZE,
                    (x + 1) * PIXEL_SIZE,
                    (y + 1) * PIXEL_SIZE
                };
                int gray = (int)input_data[y * GRID_SIZE + x];
                if (gray > 255) gray = 255;
                if (gray < 0) gray = 0;
        
                HBRUSH brush = CreateSolidBrush(RGB(gray, gray, gray));
                FillRect(hdc, &r, brush);
                DeleteObject(brush);
            }
        }
        EndPaint(hwnd, &ps);
    }
    break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}