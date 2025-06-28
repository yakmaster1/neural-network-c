#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "alglib.h"

#define E 2.718281f

// Gibt einen gleichverteilten Zufallswert im Intervall [min, max] zurück.
// Diese Funktion wird typischerweise für Xavier-Glorot-Initialisierung verwendet,
// bei der Gewichte zufällig innerhalb eines bestimmten Bereichs initialisiert werden,
// um eine stabile Varianz über Schichten eines neuronalen Netzes hinweg zu gewährleisten.
float random_uniform(float min, float max)
{
    // Erzeuge einen zufälligen Wert zwischen 0 und 1
    float normalized = (float)rand() / (float)RAND_MAX;

    // Skaliere den Wert in den Bereich [min, max] und gib ihn zurück
    return min + normalized * (max - min);
}

// Hilfsfunktion
// Gibt Leerzeichen aus, um ein Element rechtsbündig auszurichten.
// Wird für das bündige Ausgeben von Matrizen verwendet. Wird im finalen Code
// nicht verwendet.
void print_offset(int biggest_offset, int element_offset)
{
    for (int i = 0; i < biggest_offset-element_offset; i++) {printf(" ");}
}

// Helper function
// Gibt die Anzahl der Stellen (inkl. Vorzeichen) zurück, 
// die benötigt werden, um eine Float-Zahl als Ganzzahlteil darzustellen.
// Wird für das bündige Ausgeben von Matrizen verwendet. Wird im finalen Code
// nicht verwendet.
int get_size_of_float(float val)
{
    if (val == 0.0f) {return signbit(val) ? 2 : 1;}     // 0 = return 1 oder -0 = return 2
    int digits = (int)log10f(fabsf(val)) + 1;           // Anzahl Ziffern vor dem Komma
    if (digits < 1) digits = 1;                         
    if (val < 0.0f) {digits += 1;}                      // Platz für negatives Vorzeichen
    return digits;
}

// Funktion zur Erstellung und Initialisierung eines Vektors
Vector *create_v(int size, float *elements, VectorDeclaration value_declaration)
{
    if (size < 1) {printf("create_v -> 1\n"); return NULL;}     // Sicherheitsprüfung
    
    // Speicher für Vektorstruktur allokieren
    Vector *vector = malloc(sizeof(Vector));
    if (vector == NULL) {printf("Mem alloc failed\n"); return NULL;}
    
    // Speicher für die Elemente des Vektors reservieren und mit Nullen initialisieren
    vector->size = size;
    vector->elements = calloc(size, sizeof(float));
    if (vector->elements == NULL) {printf("Mem alloc failed\n"); return NULL;}

    // Initialisierungslogik basierend auf dem übergebenen value_declaration
    if (value_declaration == INIT) {
        // Benutzerdefinierte Werte übernehmen
        for (int i = 0; i < size; i++) {vector->elements[i] = elements[i];}
    }
    else if (value_declaration == RAND) {
        // Zufallswerte generieren in interval [0,1]
        for (int i = 0; i < size; i++) {vector->elements[i] = random_uniform(0,1);}
    }
    // ZERO -> Calloc
    return vector;
}

// Gibt den Speicher eines Vektors frei
void dispose_v(Vector *vector)
{
    if (vector == NULL) {printf("dispose_v -> 1\n"); return;}       // Sicherheitsprüfung
    
    free(vector->elements);
    free(vector);
}

// Gibt die Inhalte eines Vektors formatiert aus.
// Wird im finalen Code nicht verwendet.
void print_v(Vector *vector)
{
    int biggest_offset = 0;
    int total_offset = 0;
    
    // Hilfsarray zur Speicherung der individuellen Offsets je Vektorelement
    int *element_offsets = calloc(vector->size, sizeof(int));
    if (element_offsets == NULL) {printf("Mem alloc failed\n"); return;}

    // Berechne den notwendigen Versatz für jedes Element zur Formatierung
    for (int i = 0; i < vector->size; i++) {
        float val = vector->elements[i];
        int element_offset = 1;
        
        // Falls Wert ungleich Null (mit Toleranz), berechne Stellen vor dem Komma
        if (fabsf(val) > 1e-6f) {
            int digits_before_dot = (val == 0.0f) ? 1 : (int)log10f(fabsf(val)) + 1;
            // +1 für Dezimalpunkt, +2 für zwei Nachkommastellen, +1 wenn negativ
            total_offset = digits_before_dot + 1 + 2 + (val < 0.0f ? 1 : 0);
        }
        // Erhöhe Offset für negative Zahlen wegen '-' Zeichen
        if (vector->elements[i] < 0.0f) {element_offset++;}
        // Bestimme den größten Offset zur späteren Ausrichtung
        if (element_offset > biggest_offset) {biggest_offset = element_offset;}
        
        element_offsets[i] = element_offset;
    }
    
    // Zweite Schleife: Ausgabe der Vektorelemente mit korrektem Abstand
    for (int i = 0; i < vector->size; i++)
    {
        printf("|");
        
        print_offset(biggest_offset, total_offset);     // Drucke Leerzeichen zur Ausrichtung
        printf(" %.2f ", vector->elements[i]);          // Drucke das Vektorelement mit zwei Nachkommastellen
        printf("|");
        if (i < vector->size -1) {printf("\n");}        // Zeilenumbruch, wenn nicht letztes Element
    }
    free(element_offsets);
}

// Gibt mehrere Vektoren nebeneinander zeilenweise aus – mit optionalem Abstand und Trennzeichen.
// Wird im finalen Code nicht verwendet.
void printmultiple_v(int amount, Vector **vectors, int space, bool use_divider)
{
    if (amount < 1) { printf("printmultiple_v -> 1\n"); return; }   // Sicherheitsprüfung
    if (space < 0) { printf("printmultiple_v -> 2\n"); return; }    // Sicherheitsprüfung
    
    // Überprüfe, ob alle Vektoren die gleiche Größe haben
    int size = vectors[0]->size;
    for (int i = 0; i < amount; i++) 
    {
        if (vectors[i]->size != size) {printf("printmultiple_v -> 3\n"); return;}
    }

    // Standardformatierung: Feldbreite und Nachkommastellen
    int field_width = 7;
    int precision = 2;
    int added_offset = 0;
    
    // Analysiere alle Vektoren, um maximal benötigte Breite (offset) zu berechnen
    for (int a = 0; a < amount; a++)
    {
        for (int s = 0; s < vectors[a]->size; s++)
        {
            int float_size = get_size_of_float(vectors[a]->elements[s]);
            if (float_size >= added_offset) {added_offset = float_size;}
        }
    }
    
    // Hauptausgabe: Zeilenweise Ausgabe der Elemente
    for (int s = 0; s < size; s++) 
    {
        printf("|");
        for (int a = 0; a < amount; a++) 
        {
            // Abstand zwischen den Vektoren (mit optionalen Trennlinien)
            if (a != 0) 
            {
                if (use_divider) printf("|");
                for (int i = 0; i < space; i++) printf(" ");
                if (use_divider) printf("|");
            }
            // Ausgabe des Vektorelements mit dynamischer Breite
            printf(" %*.*f ", field_width + added_offset - 4, precision, vectors[a]->elements[s]);
        }
        printf("|");
        if (s < size - 1) printf("\n");     // Kein Zeilenumbruch nach letzter Zeile
    }
}

// Erstellt eine Matrix mit gegebener Zeilen- und Spaltenanzahl
// Die Matrix besteht aus Spalten-Vektoren, die je nach value_declaration initialisiert werden
Matrix *create_m(int rows, int columns, VectorDeclaration value_declaration)
{
    if (rows < 0 || columns < 0) {printf("create_m -> 1\n"); return NULL;}  // Sicherheitsprüfung
    
    // Speicher für die Matrixstruktur allokieren
    Matrix *matrix = malloc(sizeof(Matrix));
    if(matrix == NULL) {printf("Mem alloc failed\n"); return NULL;}

    matrix->rows = rows;
    matrix->columns = columns;
    
    // Speicher für die Spaltenvektoren (Array von Vector-Pointern) allokieren
    matrix->vectors = calloc(columns, sizeof(Vector*));
    if(matrix->vectors == NULL) {printf("Mem alloc failed\n"); return NULL;}

    // Sonderbehandlung: bei INIT (value_declaration == 2) wird stattdessen ZERO verwendet
    // (Da wir kein echtes Elementarray übergeben können)
    VectorDeclaration declaration = (value_declaration == 2) ? 0 : value_declaration;

    // Erzeuge für jede Spalte einen Vektor mit "rows" vielen Elementen
    for (int i = 0; i < columns; i++)
    {
        // Übergabe eines Dummy-Array "(float[]){0}" ist nötig, da nur ZERO oder RAND
        // gültige value_declaration werte sind 
        matrix->vectors[i] = create_v(rows, (float[]){0}, declaration);
    }
    return matrix;    
}

// Matrix-Print Ausgabe ist die Ausgabe mehrerer Vektoren nebeneinander.
// Wird im finalen Code nicht verwendet.
void print_m(Matrix* matrix)
{
    printmultiple_v(matrix->columns, matrix->vectors, 1, false);
}

// Gibt den Speicher einer Matrix frei
void dispose_m(Matrix *matrix)
{
    if(matrix == NULL) {printf("dispose_m -> 1\n"); return;}    // Sicherheitsprüfung
    
    for (int i = 0; i < matrix->columns; i++)
    {
        dispose_v(matrix->vectors[i]);
    }
    free(matrix->vectors);
    free(matrix);    
}

// Führt eine lineare Transformation auf einem Vektor durch:
// result = transformation * vector
void transform_linear(Matrix *transformation, Vector *vector, Vector *result)
{
    if (vector->size != transformation->columns) {printf("transform_linear -> 1\n"); return;}   // Sicherheitsprüfung
    if (result->size != transformation->rows) {printf("transform_linear -> 2\n"); return;}      // Sicherheitsprüfung
    
    
    for (int r = 0; r < transformation->rows; r++)
    {
        result->elements[r] = 0.0f;
        
        // Berechne Skalarprodukt der r-ten Zeile der Matrix mit dem Eingabevektor
        for (int c = 0; c < transformation->columns; c++)
        {
            result->elements[r] += transformation->vectors[c]->elements[r] * vector->elements[c];
        }
    }
}

// Setzt alle Werte eines Vektors auf die übergebenen Daten
void setvalues_v(Vector *vector, float *data, int size)
{
    if (size != vector->size) {printf("setvalues_v -> 1\n"); return;}   // Sicherheitsprüfung
    
    // Kopiere die Werte aus dem übergebenen Float-Array in den Vektor
    for (int i = 0; i < size; i++)
    {
        vector->elements[i] = data[i];
    }
}

// Definition: Sigmoid Funktion
float sigmoid(float input)
{
	return 1.0f / (1.0f + pow(E, -input));
}

// Definition: Ableitung Sigmoid Funktion
float abl_sigmoid(float input)
{
	return sigmoid(input) * (1.0f - sigmoid(input));
}

// Definition: ReLU Funktion
float relu(float input) 
{
    return input > 0.0f ? input : 0.0f;
}

// Definition: Ableitung ReLU Funktion
float abl_relu(float input) 
{
    return input > 0.0f ? 1.0f : 0.0f;
}

// Erstellt einen Vektor mit Werten, die zufällig im Bereich [-limit, limit] liegen
Vector *pre_init_vector(int size, float limit)
{
    if (size < 1) {printf("init_vector_xavier -> 1\n"); return NULL;}       // Sicherheitsprüfung
    
    // Allokiere Speicher für die Vektorstruktur
    Vector *vector = malloc(sizeof(Vector));
    if(vector == NULL) {printf("Mem alloc failed\n"); return NULL;}
    
    vector->size = size;
    vector->elements = calloc(size, sizeof(float));
    if(vector->elements == NULL) {printf("Mem alloc failed\n"); return NULL;}

    // Fülle Vektor mit Zufallswerten im Bereich [-limit, limit]
    for (int i = 0; i < size; i++) {
        float random_value = random_uniform(-limit, limit);
        vector->elements[i] = random_value;
    }
    return vector;
}

// Erstellt eine Matrix mit zufälligen Werten, initialisiert nach einem bestimmten Verfahren
// Hier aktuell: He-Initialisierung (empfohlen für ReLU)
// Alternative: Xavier-Glorot-Initialisierung (für Sigmoid)
Matrix *pre_init_matrix(int rows, int columns)
{
    if (rows < 0 || columns < 0) {printf("init_matrix_xavier -> 1\n"); return NULL;}    // Sicherheitsprüfung
    
    // Speicher für Matrixstruktur allokieren
    Matrix *matrix = malloc(sizeof(Matrix));
    if(matrix == NULL) {printf("Mem alloc failed\n"); return NULL;}

    matrix->rows = rows;
    matrix->columns = columns;
    matrix->vectors = calloc(columns, sizeof(Vector*));
    if(matrix->vectors == NULL) {printf("Mem alloc failed\n"); return NULL;}

    // Initialisierungsgrenze bestimmen
    // Für Sigmoid: Xavier-Glorot --- float limit = 1.0f / sqrtf((float)columns);
    
    // Für ReLU: He-Initialisierung:
    float limit = sqrtf(2.0f / columns);
    for (int i = 0; i < columns; i++)
    {
        matrix->vectors[i] = pre_init_vector(rows, limit);
    }
    return matrix;
}