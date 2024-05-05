---
layout: post
title: "Shared Behavior Patterns in Rust and Go"
date: "2023-11-11"
categories: programming-languages
---

Shared Behavior Patterns are mechanisms defined in programming languages that allow
developers to promote code re-usability, consistency, and maintainability within projects.
They also can serve as a layer of abstraction, providing a high-level reasoning between
data types that are pertinent to your own program's domain.

Without delving too deep into type theory, let us explore the motivations behind these patterns through
Go's interfaces and Rust's traits.

## Motivations for Shared Behavior Patterns

To better understand why they are so beneficial, let us look at a simple example in C.

Assume we want to build a geometry graphics library for drawing various shapes onto the screen.
We might have defined our data types like so:

```c
typedef struct {
    double x, y;
} Point2D;

typedef enum {
    SHAPE_CIRCLE,
    SHAPE_POLYGON,
} ShapeType;

typedef struct {
    ShapeType type;
} Shape;

typedef struct {
    ShapeType type;
    Point2D center;
    double radius;
} Circle;

typedef struct {
    ShapeType type;
    unsigned int nvert;
    Point *vertices;
} Polygon;
```

## Comparing Interfaces and Traits

Now that we have seen the motivations of having such a construct within a programming language,
let us port our C code to Go and Rust.

```go
type Shape interface {
    Area() float64
}

type Point2D struct {
    x, y float64
}

type Circle struct {
    center Point2D
    radius float64
}

type Polygon struct {
    vertices []Point2D
}
```

And in Rust:

```rust
trait Shape {
    fn area(&self) -> f64;
}

struct Point2D {
    x: f64,
    y: f64,
}

struct Circle {
    center: Point2D,
    radius: f64,
}

struct Polygon {
    vertices: Vec<Point2D>,
}
```

Go's interfaces use a form of structural typing, often referred to as
[*duck typing*](https://en.wikipedia.org/wiki/Duck_typing).

> If it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck.

In essence, if a type satisfies the methods provided in an interface, it is considered to implement such
interface, even when it wasn't explicitly declared to do so. This notion adheres to one of Go's key virtues,
that being simplicity. However, this is wholly at odds with the spirit behind Rust's verbosity.

In Rust, a struct must have a concrete statement of implementation through an impl block.
