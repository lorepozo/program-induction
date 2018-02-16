use polytype::{self, Type};

use std::collections::{HashMap, VecDeque};
use std::fmt;

/*
 * ERRORS
 */

#[derive(Debug, Clone)]
pub enum InferenceError {
    BadPrimitive(String),
    BadInvented(u64),
    Unify(polytype::UnificationError),
}
impl From<polytype::UnificationError> for InferenceError {
    fn from(err: polytype::UnificationError) -> Self {
        InferenceError::Unify(err)
    }
}
impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &InferenceError::BadPrimitive(ref s) => write!(f, "invalid primitive: '{}'", s),
            &InferenceError::BadInvented(i) => write!(f, "invalid invented: '{}'", i),
            &InferenceError::Unify(ref err) => write!(f, "could not unify to infer type: {}", err),
        }
    }
}
impl ::std::error::Error for InferenceError {
    fn description(&self) -> &str {
        "could not infer type"
    }
}

#[derive(Debug, Clone)]
pub struct ParseError(u64, &'static str);
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{} at index {}", self.1, self.0)
    }
}
impl ::std::error::Error for ParseError {
    fn description(&self) -> &str {
        "could not parse expression"
    }
}

/*
 * /ERRORS
 */

/// An expression context is effectively a registry for primitive and invented expressions.
///
/// Most expressions don't make sense without a context.
#[derive(Debug, Clone)]
pub struct Context<'a, 'c: 'a> {
    primitives: HashMap<&'c str, Type>,
    invented: Vec<(Variant<'a>, Type)>,
}
impl<'a, 'c> Context<'a, 'c> {
    pub fn new(primitives: HashMap<&'c str, Type>, invented: Vec<(Variant<'a>, Type)>) -> Self {
        Context {
            primitives,
            invented,
        }
    }
    pub fn invent(&mut self, expr: Variant<'a>) -> u64 {
        let mut tctx = polytype::Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        let tp = expr.infer_internal(&self, &mut tctx, &env, &mut indices)
            .expect("invalid invention");
        self.invented.push((expr, tp));
        (self.invented.len() - 1) as u64
    }
}

#[derive(Debug, Clone)]
pub enum Variant<'a> {
    Primitive(&'a str),
    Application(Box<Variant<'a>>, Box<Variant<'a>>),
    Abstraction(Box<Variant<'a>>),
    Index(u64),

    Invented(u64),
}
impl<'a> Variant<'a> {
    fn infer_internal<'c: 'a>(
        &self,
        ctx: &Context<'a, 'c>,
        mut tctx: &mut polytype::Context,
        env: &VecDeque<Type>,
        indices: &mut HashMap<u64, Type>,
    ) -> Result<Type, InferenceError> {
        match self {
            &Variant::Primitive(name) => if let Some(tp) = ctx.primitives.get(name) {
                Ok(tp.instantiate_indep(tctx))
            } else {
                Err(InferenceError::BadPrimitive(String::from(name)))
            },
            &Variant::Application(ref f, ref x) => {
                let f_tp = f.infer_internal(ctx, &mut tctx, env, indices)?;
                let x_tp = x.infer_internal(ctx, &mut tctx, env, indices)?;
                let ret_tp = tctx.new_variable();
                tctx.unify(&f_tp, &arrow![x_tp, ret_tp.clone()])?;
                Ok(ret_tp.apply(tctx))
            }
            &Variant::Abstraction(ref body) => {
                let arg_tp = tctx.new_variable();
                let mut env = env.clone();
                env.push_front(arg_tp.clone());
                let ret_tp = body.infer_internal(ctx, &mut tctx, &env, indices)?;
                Ok(arrow![arg_tp, ret_tp].apply(tctx))
            }
            &Variant::Index(i) => {
                if (i as usize) < env.len() {
                    Ok(env[i as usize].apply(tctx))
                } else {
                    Ok(indices
                        .entry(i - (env.len() as u64))
                        .or_insert_with(|| tctx.new_variable())
                        .apply(tctx))
                }
            }
            &Variant::Invented(num) => {
                if let Some(inv) = ctx.invented.get(num as usize) {
                    Ok(inv.1.instantiate_indep(tctx))
                } else {
                    Err(InferenceError::BadInvented(num))
                }
            }
        }
    }
    fn show<'c: 'a>(&self, ctx: &Context<'a, 'c>, is_function: bool) -> String {
        match self {
            &Variant::Primitive(name) => String::from(name),
            &Variant::Application(ref f, ref x) => if is_function {
                format!("{} {}", f.show(ctx, true), x.show(ctx, false))
            } else {
                format!("({} {})", f.show(ctx, true), x.show(ctx, false))
            },
            &Variant::Abstraction(ref body) => format!("(λ {})", body.show(ctx, false)),
            &Variant::Index(i) => format!("${}", i),
            &Variant::Invented(num) => {
                format!("#{}", ctx.invented[num as usize].0.show(ctx, false))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expression<'a, 'c: 'a> {
    ctx: &'a Context<'a, 'c>,
    variant: Variant<'a>,
}
impl<'a, 'c> Expression<'a, 'c> {
    pub fn infer(&self) -> Result<Type, InferenceError> {
        let mut tctx = polytype::Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        self.variant
            .infer_internal(self.ctx, &mut tctx, &env, &mut indices)
    }
    fn show(&self, is_function: bool) -> String {
        self.variant.show(self.ctx, is_function)
    }
}
impl<'a, 'c> fmt::Display for Expression<'a, 'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.show(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer() {
        let ctx = Context::new(
            hashmap![
                "singleton" => arrow![tp!(0), tp!(list(tp!(0)))],
                ">=" => arrow![tp!(int), tp!(int), tp!(bool)],
                "+" => arrow![tp!(int), tp!(int), tp!(int)],
                "0" => tp!(int),
                "1" => tp!(int),
            ],
            vec![
                (
                    Variant::Application(
                        Box::new(Variant::Primitive("+")),
                        Box::new(Variant::Primitive("1")),
                    ),
                    arrow![tp!(int), tp!(int)],
                ),
            ],
        );
        let v = Variant::Application(
            Box::new(Variant::Primitive("singleton")),
            Box::new(Variant::Application(
                Box::new(Variant::Abstraction(Box::new(Variant::Application(
                    Box::new(Variant::Application(
                        Box::new(Variant::Primitive(">=")),
                        Box::new(Variant::Index(0)),
                    )),
                    Box::new(Variant::Primitive("1")),
                )))),
                Box::new(Variant::Application(
                    Box::new(Variant::Invented(0)),
                    Box::new(Variant::Primitive("0")),
                )),
            )),
        );
        let expr = Expression {
            ctx: &ctx,
            variant: v,
        };
        assert_eq!(expr.infer().unwrap(), tp!(list(tp!(bool))));
        assert_eq!(
            format!("{}", expr),
            "(singleton ((λ (>= $0 1)) (#(+ 1) 0)))"
        );
    }
}
