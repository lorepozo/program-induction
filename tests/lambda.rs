#[macro_use]
extern crate polytype;
extern crate programinduction;

use programinduction::lambda::*;

#[test]
fn lambda_expression_parse_primitive() {
    let dsl = Language::uniform(vec![("singleton", arrow![tp!(0), tp!(list(tp!(0)))])]);
    let expr = dsl.parse("singleton").unwrap();
    assert_eq!(expr, Expression::Primitive(0));

    assert!(dsl.parse("something_else").is_err());
    assert!(dsl.parse("singleton singleton").is_err());
}

#[test]
fn lambda_expression_parse_application() {
    let dsl = Language::uniform(vec![
        ("singleton", arrow![tp!(0), tp!(list(tp!(0)))]),
        ("thing", arrow![tp!(int), tp!(int)]),
    ]);
    assert_eq!(
        dsl.parse("(singleton singleton)").unwrap(),
        Expression::Application(
            Box::new(Expression::Primitive(0)),
            Box::new(Expression::Primitive(0)),
        )
    );

    // not a valid type, but that's not a guarantee the parser makes.
    assert_eq!(
        dsl.parse("(singleton thing singleton (singleton thing))")
            .unwrap(),
        Expression::Application(
            Box::new(Expression::Application(
                Box::new(Expression::Application(
                    Box::new(Expression::Primitive(0)),
                    Box::new(Expression::Primitive(1)),
                )),
                Box::new(Expression::Primitive(0)),
            )),
            Box::new(Expression::Application(
                Box::new(Expression::Primitive(0)),
                Box::new(Expression::Primitive(1)),
            )),
        )
    );

    assert!(dsl.parse("()").is_err());
}

#[test]
fn lambda_expression_parse_index() {
    let dsl = Language::uniform(vec![("singleton", arrow![tp!(0), tp!(list(tp!(0)))])]);
    assert_eq!(
        dsl.parse("(singleton $0)").unwrap(),
        Expression::Application(
            Box::new(Expression::Primitive(0)),
            Box::new(Expression::Index(0))
        )
    );

    /// an index never makes sense outside of an application or lambda body.
    assert!(dsl.parse("$0").is_err());
}

#[test]
fn lambda_expression_parse_invented() {
    let mut dsl = Language::uniform(vec![
        ("+", arrow![tp!(int), tp!(int), tp!(int)]),
        ("1", tp!(int)),
    ]);
    dsl.invent(
        Expression::Application(
            Box::new(Expression::Primitive(0)),
            Box::new(Expression::Primitive(1)),
        ),
        0f64,
    ).unwrap();
    assert_eq!(
        dsl.parse("(#(+ 1) 1)").unwrap(),
        Expression::Application(
            Box::new(Expression::Invented(0)),
            Box::new(Expression::Primitive(1)),
        )
    );
    assert!(dsl.parse("(#(+ 1 1) 1)").is_err());
}

#[test]
fn lambda_expression_parse_abstraction() {
    let mut dsl = Language::uniform(vec![
        ("+", arrow![tp!(int), tp!(int), tp!(int)]),
        ("1", tp!(int)),
    ]);
    dsl.invent(
        Expression::Abstraction(Box::new(Expression::Application(
            Box::new(Expression::Application(
                Box::new(Expression::Primitive(0)),
                Box::new(Expression::Application(
                    Box::new(Expression::Application(
                        Box::new(Expression::Primitive(0)),
                        Box::new(Expression::Primitive(1)),
                    )),
                    Box::new(Expression::Primitive(1)),
                )),
            )),
            Box::new(Expression::Index(0)),
        ))),
        0f64,
    ).unwrap();
    let expr = dsl.parse("(λ (+ $0))").unwrap();
    assert_eq!(
        expr,
        Expression::Abstraction(Box::new(Expression::Application(
            Box::new(Expression::Primitive(0)),
            Box::new(Expression::Index(0)),
        )))
    );
    assert_eq!(dsl.display(&expr), "(λ (+ $0))");
    let expr = dsl.parse("(#(lambda (+ (+ 1 1) $0)) ((lambda (+ $0 1)) 1))")
        .unwrap();
    assert_eq!(
        expr,
        Expression::Application(
            Box::new(Expression::Invented(0)),
            Box::new(Expression::Application(
                Box::new(Expression::Abstraction(Box::new(Expression::Application(
                    Box::new(Expression::Application(
                        Box::new(Expression::Primitive(0)),
                        Box::new(Expression::Index(0)),
                    )),
                    Box::new(Expression::Primitive(1)),
                )))),
                Box::new(Expression::Primitive(1)),
            )),
        ),
    );
    assert_eq!(
        dsl.display(&expr),
        "(#(λ (+ (+ 1 1) $0)) ((λ (+ $0 1)) 1))"
    );
    let expr = dsl.parse("(lambda $0)").unwrap();
    assert_eq!(
        expr,
        Expression::Abstraction(Box::new(Expression::Index(0)))
    );
    assert_eq!(dsl.display(&expr), "(λ $0)");
}

#[test]
fn lambda_expression_infer() {
    let mut dsl = Language::uniform(vec![
        ("singleton", arrow![tp!(0), tp!(list(tp!(0)))]),
        (">=", arrow![tp!(int), tp!(int), tp!(bool)]),
        ("+", arrow![tp!(int), tp!(int), tp!(int)]),
        ("0", tp!(int)),
        ("1", tp!(int)),
    ]);
    dsl.invent(
        Expression::Application(
            Box::new(Expression::Primitive(2)),
            Box::new(Expression::Primitive(4)),
        ),
        0f64,
    ).unwrap();
    let expr = Expression::Application(
        Box::new(Expression::Primitive(0)),
        Box::new(Expression::Application(
            Box::new(Expression::Abstraction(Box::new(Expression::Application(
                Box::new(Expression::Application(
                    Box::new(Expression::Primitive(1)),
                    Box::new(Expression::Index(0)),
                )),
                Box::new(Expression::Primitive(4)),
            )))),
            Box::new(Expression::Application(
                Box::new(Expression::Invented(0)),
                Box::new(Expression::Primitive(3)),
            )),
        )),
    );
    assert_eq!(dsl.infer(&expr).unwrap(), tp!(list(tp!(bool))));
    assert_eq!(
        dsl.display(&expr),
        "(singleton ((λ (>= $0 1)) (#(+ 1) 0)))"
    );
}

#[test]
fn lambda_eval_simplest() {
    let dsl = Language::uniform(vec![
        ("0", tp!(int)),
        ("1", tp!(int)),
        ("+", arrow![tp!(int), tp!(int), tp!(int)]),
    ]);

    fn evaluate(primitive: &str, inps: &[i32]) -> i32 {
        match primitive {
            "0" => 0,
            "1" => 1,
            "+" => inps[0] + inps[1],
            _ => unreachable!(),
        }
    }
    let expr = dsl.parse("(+ (+ 1 1) 1)").unwrap();
    let out = dsl.eval(&expr, SimpleEvaluator::of(evaluate), &[]);
    assert_eq!(out, Some(3));
}

#[test]
fn lambda_eval_somewhat_simple() {
    let dsl = Language::uniform(vec![
        ("0", tp!(int)),
        ("1", tp!(int)),
        ("+", arrow![tp!(int), tp!(int), tp!(int)]),
        ("eq", arrow![tp!(int), tp!(int), tp!(bool)]),
        ("not", arrow![tp!(bool), tp!(bool)]),
    ]);

    #[derive(Clone, PartialEq, Debug)]
    enum ArithSpace {
        Bool(bool),
        Num(i32),
    }

    fn evaluate(primitive: &str, inps: &[ArithSpace]) -> ArithSpace {
        match primitive {
            "0" => ArithSpace::Num(0),
            "1" => ArithSpace::Num(1),
            "+" => match (&inps[0], &inps[1]) {
                (&ArithSpace::Num(x), &ArithSpace::Num(y)) => ArithSpace::Num(x + y),
                _ => unreachable!(),
            },
            "eq" => match (&inps[0], &inps[1]) {
                (&ArithSpace::Num(x), &ArithSpace::Num(y)) => ArithSpace::Bool(x == y),
                _ => unreachable!(),
            },
            "not" => match inps[0] {
                ArithSpace::Bool(b) => ArithSpace::Bool(!b),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    let expr = dsl.parse("(λ (not (eq (+ 1 $0) 1)))").unwrap();

    let out = dsl.eval(&expr, SimpleEvaluator::of(evaluate), &[ArithSpace::Num(1)]);
    assert_eq!(out, Some(ArithSpace::Bool(true)));

    let out = dsl.eval(&expr, SimpleEvaluator::of(evaluate), &[ArithSpace::Num(0)]);
    assert_eq!(out, Some(ArithSpace::Bool(false)));
}

#[test]
fn lambda_eval_firstclass() {
    #[derive(Clone)]
    enum ListSpace {
        Num(i32),
        List(Vec<i32>),
        Func(LiftedFunction<ListSpace, ListEvaluator>),
    }
    impl PartialEq for ListSpace {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (&ListSpace::Num(x), &ListSpace::Num(y)) => x == y,
                (&ListSpace::List(ref xs), &ListSpace::List(ref ys)) => xs == ys,
                _ => false,
            }
        }
    }

    #[derive(Clone)]
    struct ListEvaluator;
    impl Evaluator for ListEvaluator {
        type Space = ListSpace;
        fn evaluate(&self, primitive: &str, inps: &[Self::Space]) -> Self::Space {
            match primitive {
                "0" => ListSpace::Num(0),
                "1" => ListSpace::Num(1),
                "+" => match (&inps[0], &inps[1]) {
                    (&ListSpace::Num(x), &ListSpace::Num(y)) => ListSpace::Num(x + y),
                    _ => unreachable!(),
                },
                "singleton" => match inps[0] {
                    ListSpace::Num(x) => ListSpace::List(vec![x]),
                    _ => unreachable!(),
                },
                "chain" => match (&inps[0], &inps[1]) {
                    (&ListSpace::List(ref xs), &ListSpace::List(ref ys)) => {
                        ListSpace::List(xs.into_iter().chain(ys).cloned().collect())
                    }
                    _ => unreachable!(),
                },
                "map" => match (&inps[0], &inps[1]) {
                    (&ListSpace::Func(ref f), &ListSpace::List(ref xs)) => ListSpace::List(
                        xs.into_iter()
                            .cloned()
                            .map(|x| match f.eval(&[ListSpace::Num(x)]) {
                                ListSpace::Num(y) => y,
                                _ => panic!("map given invalid function"),
                            })
                            .collect(),
                    ),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        fn lift(&self, f: LiftedFunction<Self::Space, Self>) -> Result<Self::Space, ()> {
            Ok(ListSpace::Func(f))
        }
    }

    let dsl = Language::uniform(vec![
        ("0", tp!(int)),
        ("1", tp!(int)),
        ("+", arrow![tp!(int), tp!(int), tp!(int)]),
        ("singleton", arrow![tp!(int), tp!(intlist)]),
        ("chain", arrow![tp!(intlist), tp!(intlist), tp!(intlist)]),
        (
            "map",
            arrow![arrow![tp!(int), tp!(int)], tp!(intlist), tp!(intlist)],
        ),
    ]);
    let examples = vec![
        (
            vec![ListSpace::Num(5), ListSpace::List(vec![2, 5, 7])],
            ListSpace::List(vec![7, 10, 12]),
        ),
        (
            vec![ListSpace::Num(2), ListSpace::List(vec![1, 2, 3])],
            ListSpace::List(vec![3, 4, 5]),
        ),
    ];
    // task: add-k
    let tp = arrow![tp!(intlist), tp!(int), tp!(intlist)];
    let task = task_by_evaluation(ListEvaluator, tp, &examples);

    // good solution:
    let expr = dsl.parse("(λ (λ (map (λ (+ $0 $2)) $0)))").unwrap();
    assert!((task.oracle)(&dsl, &expr).is_finite());
    // bad solution:
    let expr = dsl.parse("(λ (λ (map (λ (+ $0 1)) $0)))").unwrap();
    assert!((task.oracle)(&dsl, &expr).is_infinite());
}
