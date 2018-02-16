use polytype::{Context, Type, UnificationError};

use std::collections::{HashMap, VecDeque};
use std::fmt;

#[derive(Debug, Clone)]
pub struct InferenceError(UnificationError);
impl From<UnificationError> for InferenceError {
    fn from(err: UnificationError) -> Self {
        InferenceError(err)
    }
}
impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "InferenceError({})", self.0)
    }
}
impl ::std::error::Error for InferenceError {
    fn description(&self) -> &str {
        "could not infer type"
    }
}

#[derive(Debug, Clone)]
pub struct ParseError(u64);
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "ParseError({})", self.0)
    }
}
impl ::std::error::Error for ParseError {
    fn description(&self) -> &str {
        "could not parse expression"
    }
}

#[derive(Debug, Clone)]
pub enum Expression {
    Primitive(&'static str, Type),
    Application(Box<Expression>, Box<Expression>),
    Abstraction(Box<Expression>),
    Index(u32),

    Invented(Box<Expression>),
}
impl Expression {
    pub fn infer(&self) -> Result<Type, InferenceError> {
        let mut ctx = Context::default();
        let env = VecDeque::new();
        let mut indices = HashMap::new();
        self.infer_internal(&mut ctx, &env, &mut indices)
    }
    fn infer_internal(
        &self,
        mut ctx: &mut Context,
        env: &VecDeque<Type>,
        indices: &mut HashMap<u32, Type>,
    ) -> Result<Type, InferenceError> {
        match self {
            &Expression::Primitive(_, ref tp) => Ok(tp.instantiate_indep(ctx)),
            &Expression::Application(ref f, ref x) => {
                let f_tp = f.infer_internal(&mut ctx, env, indices)?;
                let x_tp = x.infer_internal(&mut ctx, env, indices)?;
                let ret_tp = ctx.new_variable();
                ctx.unify(&f_tp, &arrow![x_tp, ret_tp.clone()])?;
                Ok(ret_tp.apply(ctx))
            }
            &Expression::Abstraction(ref body) => {
                let arg_tp = ctx.new_variable();
                let mut env = env.clone();
                env.push_front(arg_tp.clone());
                let ret_tp = body.infer_internal(&mut ctx, &env, indices)?;
                Ok(arrow![arg_tp, ret_tp].apply(ctx))
            }
            &Expression::Index(i) => {
                if (i as usize) < env.len() {
                    Ok(env[i as usize].apply(ctx))
                } else {
                    Ok(indices
                        .entry(i - (env.len() as u32))
                        .or_insert_with(|| ctx.new_variable())
                        .apply(ctx))
                }
            }
            &Expression::Invented(ref expr) => Ok(expr.infer()?.instantiate_indep(ctx)),
        }
    }
    fn show(&self, is_function: bool) -> String {
        match self {
            &Expression::Primitive(name, _) => String::from(name),
            &Expression::Application(ref f, ref x) => if is_function {
                format!("{} {}", f.show(true), x.show(false))
            } else {
                format!("({} {})", f.show(true), x.show(false))
            },
            &Expression::Abstraction(ref body) => format!("(λ {})", body.show(false)),
            &Expression::Index(i) => format!("{}", i),
            &Expression::Invented(ref expr) => format!("#{}", expr.show(false)),
        }
    }
}
impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.show(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_infer() {
        let expr = Expression::Application(
            Box::new(Expression::Primitive(
                "singleton",
                arrow![tp!(0), tp!(list(tp!(0)))],
            )),
            Box::new(Expression::Application(
                Box::new(Expression::Abstraction(Box::new(Expression::Application(
                    Box::new(Expression::Application(
                        Box::new(Expression::Primitive(
                            ">",
                            arrow![tp!(int), tp!(int), tp!(bool)],
                        )),
                        Box::new(Expression::Index(0)),
                    )),
                    Box::new(Expression::Primitive("1", tp!(int))),
                )))),
                Box::new(Expression::Primitive("0", tp!(int))),
            )),
        );
        assert_eq!(expr.infer().unwrap(), tp!(list(tp!(bool))),);
    }
}
