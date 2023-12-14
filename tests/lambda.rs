use polytype::{ptp, tp};
use programinduction::{lambda::*, Task};

#[test]
fn lambda_expression_parse_primitive() {
    let dsl = Language::uniform(vec![(
        "singleton",
        ptp!(0; @arrow[tp!(0), tp!(list(tp!(0)))]),
    )]);
    let expr = dsl.parse("singleton").unwrap();
    assert_eq!(expr, Expression::Primitive(0));

    assert!(dsl.parse("something_else").is_err());
    assert!(dsl.parse("singleton singleton").is_err());
}

#[test]
fn lambda_expression_parse_application() {
    let dsl = Language::uniform(vec![
        ("singleton", ptp!(0; @arrow[tp!(0), tp!(list(tp!(0)))])),
        ("thing", ptp!(@arrow[tp!(int), tp!(int)])),
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
    let dsl = Language::uniform(vec![(
        "singleton",
        ptp!(0; @arrow[tp!(0), tp!(list(tp!(0)))]),
    )]);
    assert_eq!(
        dsl.parse("(singleton $0)").unwrap(),
        Expression::Application(
            Box::new(Expression::Primitive(0)),
            Box::new(Expression::Index(0))
        )
    );

    // an index never makes sense outside of an application or lambda body.
    assert!(dsl.parse("$0").is_err());
}

#[test]
fn lambda_expression_parse_invented() {
    let mut dsl = Language::uniform(vec![
        ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
        ("1", ptp!(int)),
    ]);
    dsl.invent(
        Expression::Application(
            Box::new(Expression::Primitive(0)),
            Box::new(Expression::Primitive(1)),
        ),
        0f64,
    )
    .unwrap();
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
        ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
        ("1", ptp!(int)),
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
    )
    .unwrap();
    let expr = dsl.parse("(λ (+ $0))").unwrap();
    assert_eq!(
        expr,
        Expression::Abstraction(Box::new(Expression::Application(
            Box::new(Expression::Primitive(0)),
            Box::new(Expression::Index(0)),
        )))
    );
    assert_eq!(dsl.display(&expr), "(λ (+ $0))");
    let expr = dsl
        .parse("(#(lambda (+ (+ 1 1) $0)) ((lambda (+ $0 1)) 1))")
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
    assert_eq!(dsl.display(&expr), "(#(λ (+ (+ 1 1) $0)) ((λ (+ $0 1)) 1))");
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
        ("singleton", ptp!(0; @arrow[tp!(0), tp!(list(tp!(0)))])),
        (">=", ptp!(@arrow[tp!(int), tp!(int), tp!(bool)])),
        ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
        ("0", ptp!(int)),
        ("1", ptp!(int)),
    ]);
    dsl.invent(
        Expression::Application(
            Box::new(Expression::Primitive(2)),
            Box::new(Expression::Primitive(4)),
        ),
        0f64,
    )
    .unwrap();
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
    assert_eq!(dsl.infer(&expr).unwrap(), ptp!(list(tp!(bool))));
    assert_eq!(dsl.display(&expr), "(singleton ((λ (>= $0 1)) (#(+ 1) 0)))");
}

#[test]
fn lambda_eval_simplest() {
    let dsl = Language::uniform(vec![
        ("0", ptp!(int)),
        ("1", ptp!(int)),
        ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
    ]);

    fn evaluate(primitive: &str, inps: &[i32]) -> Result<i32, ()> {
        match primitive {
            "0" => Ok(0),
            "1" => Ok(1),
            "+" => Ok(inps[0] + inps[1]),
            _ => unreachable!(),
        }
    }
    let expr = dsl.parse("(+ (+ 1 1) 1)").unwrap();
    let out = dsl.eval(&expr, SimpleEvaluator::from(evaluate), &[]);
    assert_eq!(out, Ok(3));
}

#[test]
fn lambda_eval_somewhat_simple() {
    let dsl = Language::uniform(vec![
        ("0", ptp!(int)),
        ("1", ptp!(int)),
        ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
        ("eq", ptp!(@arrow[tp!(int), tp!(int), tp!(bool)])),
        ("not", ptp!(@arrow[tp!(bool), tp!(bool)])),
    ]);

    #[derive(Clone, PartialEq, Debug)]
    enum ArithSpace {
        Bool(bool),
        Num(i32),
    }

    fn evaluate(primitive: &str, inps: &[ArithSpace]) -> Result<ArithSpace, ()> {
        match primitive {
            "0" => Ok(ArithSpace::Num(0)),
            "1" => Ok(ArithSpace::Num(1)),
            "+" => match (&inps[0], &inps[1]) {
                (&ArithSpace::Num(x), &ArithSpace::Num(y)) => Ok(ArithSpace::Num(x + y)),
                _ => unreachable!(),
            },
            "eq" => match (&inps[0], &inps[1]) {
                (&ArithSpace::Num(x), &ArithSpace::Num(y)) => Ok(ArithSpace::Bool(x == y)),
                _ => unreachable!(),
            },
            "not" => match inps[0] {
                ArithSpace::Bool(b) => Ok(ArithSpace::Bool(!b)),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    let expr = dsl.parse("(λ (not (eq (+ 1 $0) 1)))").unwrap();

    let out = dsl.eval(&expr, SimpleEvaluator::from(evaluate), &[ArithSpace::Num(1)]);
    assert_eq!(out, Ok(ArithSpace::Bool(true)));

    let out = dsl.eval(&expr, SimpleEvaluator::from(evaluate), &[ArithSpace::Num(0)]);
    assert_eq!(out, Ok(ArithSpace::Bool(false)));
}

#[test]
fn lambda_eval_firstclass() {
    #[derive(Clone, PartialEq)]
    enum ListSpace {
        Num(i32),
        List(Vec<i32>),
        Func(LiftedFunction<ListSpace, ListEvaluator>),
    }

    #[derive(Clone)]
    struct ListEvaluator;
    impl Evaluator for ListEvaluator {
        type Space = ListSpace;
        type Error = ();
        fn evaluate(
            &self,
            primitive: &str,
            inps: &[Self::Space],
        ) -> Result<Self::Space, Self::Error> {
            match primitive {
                "0" => Ok(ListSpace::Num(0)),
                "1" => Ok(ListSpace::Num(1)),
                "+" => match (&inps[0], &inps[1]) {
                    (&ListSpace::Num(x), &ListSpace::Num(y)) => Ok(ListSpace::Num(x + y)),
                    _ => unreachable!(),
                },
                "singleton" => match inps[0] {
                    ListSpace::Num(x) => Ok(ListSpace::List(vec![x])),
                    _ => unreachable!(),
                },
                "chain" => match (&inps[0], &inps[1]) {
                    (ListSpace::List(xs), ListSpace::List(ys)) => {
                        Ok(ListSpace::List(xs.iter().chain(ys).cloned().collect()))
                    }
                    _ => unreachable!(),
                },
                "map" => match (&inps[0], &inps[1]) {
                    (ListSpace::Func(f), ListSpace::List(xs)) => Ok(ListSpace::List(
                        xs.iter()
                            .map(|x| {
                                f.eval(&[ListSpace::Num(*x)])
                                    .map(|v| match v {
                                        ListSpace::Num(y) => y,
                                        _ => panic!("map given invalid function"),
                                    })
                                    .map_err(|_| ())
                            })
                            .collect::<Result<_, _>>()?,
                    )),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        fn lift(&self, f: LiftedFunction<Self::Space, Self>) -> Option<Self::Space> {
            Some(ListSpace::Func(f))
        }
    }

    let dsl = Language::uniform(vec![
        ("0", ptp!(int)),
        ("1", ptp!(int)),
        ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
        ("singleton", ptp!(@arrow[tp!(int), tp!(list(tp!(int)))])),
        (
            "chain",
            ptp!(@arrow[tp!(list(tp!(int))), tp!(list(tp!(int))), tp!(list(tp!(int)))]),
        ),
        (
            "map",
            ptp!(@arrow[
                tp!(@arrow[tp!(int), tp!(int)]),
                tp!(list(tp!(int))),
                tp!(list(tp!(int)))
            ]),
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
    let tp = ptp!(@arrow[tp!(list(tp!(int))), tp!(int), tp!(list(tp!(int)))]);
    let task = task_by_evaluation(ListEvaluator, tp, &examples);

    // good solution:
    let expr = dsl.parse("(λ (λ (map (λ (+ $0 $2)) $0)))").unwrap();
    assert!(task.oracle(&dsl, &expr).is_finite());
    // bad solution:
    let expr = dsl.parse("(λ (λ (map (λ (+ $0 1)) $0)))").unwrap();
    assert!(task.oracle(&dsl, &expr).is_infinite());
}

#[test]
fn lambda_lazy_eval() {
    #[derive(Clone, Debug, PartialEq)]
    struct ListError(&'static str);

    #[derive(Clone, Debug, PartialEq)]
    enum ListSpace {
        Bool(bool),
        Num(i32),
        List(Vec<i32>),
    }

    #[derive(Copy, Clone)]
    struct ListsEvaluator;
    impl LazyEvaluator for ListsEvaluator {
        type Space = ListSpace;
        type Error = ListError;
        fn lazy_evaluate(
            &self,
            primitive: &str,
            inps: &[LiftedLazyFunction<Self::Space, Self>],
        ) -> Result<Self::Space, Self::Error> {
            match primitive {
                "-1" => Ok(ListSpace::Num(-1)),
                "empty?" => match inps[0].eval(&[])? {
                    ListSpace::List(xs) => Ok(ListSpace::Bool(xs.is_empty())),
                    _ => unreachable!(),
                },
                "if" => match inps[0].eval(&[])? {
                    ListSpace::Bool(true) => inps[1].eval(&[]),
                    ListSpace::Bool(false) => inps[2].eval(&[]),
                    _ => unreachable!(),
                },
                "car" => match inps[0].eval(&[])? {
                    ListSpace::List(xs) => {
                        if !xs.is_empty() {
                            Ok(ListSpace::Num(xs[0]))
                        } else {
                            Err(ListError("cannot get car of empty list"))
                        }
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
    }

    let dsl = Language::uniform(vec![
        ("if", ptp!(0; @arrow[tp!(bool), tp!(0), tp!(0), tp!(0)])),
        ("empty?", ptp!(0; @arrow[tp!(list(tp!(0))), tp!(bool)])),
        ("car", ptp!(0; @arrow[tp!(list(tp!(0))), tp!(0)])),
        ("-1", ptp!(int)),
    ]);
    let examples = vec![
        (vec![ListSpace::List(vec![])], ListSpace::Num(-1)),
        (vec![ListSpace::List(vec![2])], ListSpace::Num(2)),
        (vec![ListSpace::List(vec![42])], ListSpace::Num(42)),
    ];
    let task = task_by_lazy_evaluation(
        ListsEvaluator,
        ptp!(@arrow[tp!(list(tp!(int))), tp!(int)]),
        &examples,
    );

    let expr = dsl.parse("(λ (if (empty? $0) -1 (car $0)))").unwrap();
    assert!(task.oracle(&dsl, &expr).is_finite());
}

#[test]
fn lambda_lazy_eval_firstclass() {
    #[derive(Clone, PartialEq)]
    enum ListSpace {
        Num(i32),
        List(Vec<i32>),
        Func(LiftedLazyFunction<ListSpace, LazyListEvaluator>),
    }

    #[derive(Clone)]
    struct LazyListEvaluator;
    impl LazyEvaluator for LazyListEvaluator {
        type Space = ListSpace;
        type Error = ();
        fn lazy_evaluate(
            &self,
            primitive: &str,
            inps: &[LiftedLazyFunction<Self::Space, Self>],
        ) -> Result<Self::Space, Self::Error> {
            match primitive {
                "0" => Ok(ListSpace::Num(0)),
                "1" => Ok(ListSpace::Num(1)),
                "+" => match (inps[0].eval(&[])?, inps[1].eval(&[])?) {
                    (ListSpace::Num(x), ListSpace::Num(y)) => Ok(ListSpace::Num(x + y)),
                    _ => unreachable!(),
                },
                "singleton" => match inps[0].eval(&[])? {
                    ListSpace::Num(x) => Ok(ListSpace::List(vec![x])),
                    _ => unreachable!(),
                },
                "chain" => match (inps[0].eval(&[])?, inps[1].eval(&[])?) {
                    (ListSpace::List(xs), ListSpace::List(ys)) => {
                        Ok(ListSpace::List(xs.into_iter().chain(ys).collect()))
                    }
                    _ => unreachable!(),
                },
                "map" => match (inps[0].eval(&[])?, inps[1].eval(&[])?) {
                    (ListSpace::Func(f), ListSpace::List(xs)) => Ok(ListSpace::List(
                        xs.into_iter()
                            .map(|x| {
                                f.eval(&[ListSpace::Num(x)])
                                    .map(|v| match v {
                                        ListSpace::Num(y) => y,
                                        _ => panic!("map given invalid function"),
                                    })
                                    .map_err(|_| ())
                            })
                            .collect::<Result<_, _>>()?,
                    )),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        fn lift(&self, f: LiftedLazyFunction<Self::Space, Self>) -> Option<Self::Space> {
            Some(ListSpace::Func(f))
        }
    }

    let dsl = Language::uniform(vec![
        ("0", ptp!(int)),
        ("1", ptp!(int)),
        ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
        ("singleton", ptp!(@arrow[tp!(int), tp!(list(tp!(int)))])),
        (
            "chain",
            ptp!(@arrow[tp!(list(tp!(int))), tp!(list(tp!(int))), tp!(list(tp!(int)))]),
        ),
        (
            "map",
            ptp!(@arrow[
                tp!(@arrow[tp!(int), tp!(int)]),
                tp!(list(tp!(int))),
                tp!(list(tp!(int)))
            ]),
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
    let tp = ptp!(@arrow[tp!(list(tp!(int))), tp!(int), tp!(list(tp!(int)))]);
    let task = task_by_lazy_evaluation(LazyListEvaluator, tp, &examples);

    // good solution:
    let expr = dsl.parse("(λ (λ (map (λ (+ $0 $2)) $0)))").unwrap();
    assert!(task.oracle(&dsl, &expr).is_finite());
    // bad solution:
    let expr = dsl.parse("(λ (λ (map (λ (+ $0 1)) $0)))").unwrap();
    assert!(task.oracle(&dsl, &expr).is_infinite());
}
