#[macro_use]
extern crate polytype;

extern crate programinduction;

use programinduction::lambda::{Expression, Language};

#[test]
fn expression_parse_primitive() {
    let dsl = Language::uniform(vec![("singleton", arrow![tp!(0), tp!(list(tp!(0)))])]);
    let expr = dsl.parse("singleton").unwrap();
    assert_eq!(expr, Expression::Primitive(0));

    assert!(dsl.parse("something_else").is_err());
    assert!(dsl.parse("singleton singleton").is_err());
}

#[test]
fn expression_parse_application() {
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
fn expression_parse_index() {
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
fn expression_parse_invented() {
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
fn expression_parse_abstraction() {
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
    assert_eq!(dsl.stringify(&expr), "(λ (+ $0))");
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
        dsl.stringify(&expr),
        "(#(λ (+ (+ 1 1) $0)) ((λ (+ $0 1)) 1))"
    );
    let expr = dsl.parse("(lambda $0)").unwrap();
    assert_eq!(
        expr,
        Expression::Abstraction(Box::new(Expression::Index(0)))
    );
    assert_eq!(dsl.stringify(&expr), "(λ $0)");
}

#[test]
fn expression_infer() {
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
        dsl.stringify(&expr),
        "(singleton ((λ (>= $0 1)) (#(+ 1) 0)))"
    );
}
