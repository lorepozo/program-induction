use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::rc::Rc;

use super::LispError;
use super::sexp::Sexp;

thread_local!(
    static BUILTINS: HashMap<&'static str, Rc<Value>> = {
        let lambda = Rc::new(Value::BuiltinSpecial(builtin::lambda));
        let eq = Rc::new(Value::Builtin(builtin::eq));
        vec![
            ("null", Rc::new(Value::Null)),
            ("if", Rc::new(Value::BuiltinSpecial(builtin::if_))),
            ("lambda", Rc::clone(&lambda)),
            ("λ", lambda),
            ("and", Rc::new(Value::BuiltinSpecial(builtin::and))),
            ("or", Rc::new(Value::BuiltinSpecial(builtin::or))),
            ("+", Rc::new(Value::Builtin(builtin::add))),
            ("-", Rc::new(Value::Builtin(builtin::sub))),
            ("not", Rc::new(Value::Builtin(builtin::not))),
            ("*", Rc::new(Value::Builtin(builtin::mul))),
            ("/", Rc::new(Value::Builtin(builtin::div))),
            ("quotient", Rc::new(Value::Builtin(builtin::quotient))),
            ("remainder", Rc::new(Value::Builtin(builtin::remainder))),
            ("modulo", Rc::new(Value::Builtin(builtin::modulo))),
            ("cons", Rc::new(Value::Builtin(builtin::cons))),
            ("car", Rc::new(Value::Builtin(builtin::car))),
            ("cdr", Rc::new(Value::Builtin(builtin::cdr))),
            ("=", Rc::clone(&eq)),
            ("equal?", Rc::clone(&eq)),
            ("eqv?", Rc::clone(&eq)),
            ("eq?", eq),
            (">", Rc::new(Value::Builtin(builtin::gt))),
            (">=", Rc::new(Value::Builtin(builtin::gte))),
            ("<", Rc::new(Value::Builtin(builtin::lt))),
            ("<=", Rc::new(Value::Builtin(builtin::lte))),
            ("!=", Rc::new(Value::Builtin(builtin::neq))),
            ("pair?", Rc::new(Value::Builtin(builtin::is_pair))),
            ("null?", Rc::new(Value::Builtin(builtin::is_null))),
            ("identity", Rc::new(Value::Builtin(builtin::identity))),
            ("list", Rc::new(Value::Builtin(builtin::list))),
            ("list-ref", Rc::new(Value::Builtin(builtin::list_ref))),
            ("append", Rc::new(Value::Builtin(builtin::append))),
            ("append-map", Rc::new(Value::Builtin(builtin::appendmap))),
            ("map", Rc::new(Value::Builtin(builtin::map))),
            ("filter", Rc::new(Value::Builtin(builtin::filter))),
            ("foldl", Rc::new(Value::Builtin(builtin::foldl))),
            ("count", Rc::new(Value::Builtin(builtin::count))),
            ("member", Rc::new(Value::Builtin(builtin::member))),
            ("string-length", Rc::new(Value::Builtin(builtin::string_length))),
            ("string-downcase", Rc::new(Value::Builtin(builtin::string_downcase))),
            ("string-upcase", Rc::new(Value::Builtin(builtin::string_upcase))),
            ("string-append", Rc::new(Value::Builtin(builtin::string_append))),
            ("substring", Rc::new(Value::Builtin(builtin::substring))),
            ("string-split", Rc::new(Value::Builtin(builtin::string_split))),
            ("string-join", Rc::new(Value::Builtin(builtin::string_join))),
        ].into_iter().collect()
    }
);

#[cfg_attr(feature = "cargo-clippy", allow(type_complexity))]
#[derive(Clone)]
pub enum Value {
    Bool(bool),
    Integer(i32),
    Real(f64),
    Char(char),
    Str(String),
    // notably not included: Ident(String),
    Pair(Rc<Value>, Rc<Value>),
    Null,

    Lambda(Vec<String>, Rc<Sexp>),
    BuiltinSpecial(fn(VecDeque<Sexp>, &mut Env) -> Result<Rc<Value>, LispError>),
    Builtin(fn(VecDeque<Rc<Value>>, &mut Env) -> Result<Rc<Value>, LispError>),
}
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Value::Bool(b) => write!(f, "Bool({:?})", b),
            Value::Integer(i) => write!(f, "Integer({:?})", i),
            Value::Real(r) => write!(f, "Real({:?})", r),
            Value::Char(ref c) => write!(f, "Char({:?})", c),
            Value::Str(ref s) => write!(f, "Str({:?})", s),
            Value::Pair(ref l, ref r) => write!(f, "Pair({:?}, {:?})", l, r),
            Value::Null => write!(f, "Null"),
            Value::Lambda(ref params, ref body) => write!(f, "Lambda({:?}, {:?})", params, body),
            Value::Builtin(_) => write!(f, "Builtin(..)"),
            Value::BuiltinSpecial(_) => write!(f, "BuiltinSpecial(..)"),
        }
    }
}
impl From<Sexp> for Value {
    fn from(sexp: Sexp) -> Value {
        match sexp {
            Sexp::Bool(v) => Value::Bool(v),
            Sexp::Integer(v) => Value::Integer(v),
            Sexp::Real(v) => Value::Real(v),
            Sexp::Char(v) => Value::Char(v),
            Sexp::Str(v) => Value::Str(v),
            Sexp::Pair(l, r) => Value::Pair(Rc::new(Value::from(*l)), Rc::new(Value::from(*r))),
            Sexp::Null => Value::Null,
            Sexp::Ident(_) => panic!("ident is not a valid value"),
        }
    }
}

/// the environment is a list of assignments to identifiers.
pub struct Env(Vec<(String, Rc<Value>)>);
impl Env {
    fn get(&self, s: &str) -> Result<Rc<Value>, LispError> {
        self.0
            .iter()
            .rev()
            .find(|&&(ref l, _)| l == s)
            .map(|&(_, ref v)| Rc::clone(v))
            .or_else(|| BUILTINS.with(|h| h.get(s).map(|x| Rc::clone(x))))
            .ok_or_else(|| LispError::IdentNotFound(String::from(s)))
    }
}

pub fn eval(sexp: &Sexp) -> Result<Rc<Value>, LispError> {
    eval_sexp(sexp, &mut Env(Vec::new()))
}

fn eval_sexp(sexp: &Sexp, env: &mut Env) -> Result<Rc<Value>, LispError> {
    match *sexp {
        Sexp::Pair(ref car, ref cdr) => {
            let mut xs = VecDeque::new();
            read_list(cdr, env, &mut xs);
            xs.pop_back(); // no null
            let f = eval_sexp(car, env)?;
            eval_call(&f, xs, env)
        }
        Sexp::Ident(ref s) => env.get(s),
        _ => Ok(Rc::new(Value::from(sexp.clone()))),
    }
}

fn eval_call(f: &Rc<Value>, xs: VecDeque<Sexp>, env: &mut Env) -> Result<Rc<Value>, LispError> {
    match **f {
        Value::BuiltinSpecial(f) => f(xs, env),
        _ => {
            let xs = xs.iter()
                .map(|s| eval_sexp(s, env))
                .collect::<Result<VecDeque<_>, LispError>>()?;
            eval_call_nonspecial(f, xs, env)
        }
    }
}

/// like `eval_call`, but takes already-evaluated expression
fn eval_call_nonspecial(
    f: &Rc<Value>,
    xs: VecDeque<Rc<Value>>,
    env: &mut Env,
) -> Result<Rc<Value>, LispError> {
    match **f {
        Value::Lambda(ref params, ref body) => {
            let original_env_len = env.0.len();
            for (s, v) in params.iter().zip(xs) {
                env.0.push((s.clone(), v))
            }
            let res = eval_sexp(body, env);
            env.0.truncate(original_env_len);
            res
        }
        Value::Builtin(f) => f(xs, env),
        _ => Err(LispError::InvalidCall),
    }
}

fn read_list(sexp: &Sexp, env: &mut Env, v: &mut VecDeque<Sexp>) {
    match *sexp {
        Sexp::Pair(ref car, ref cdr) => {
            v.push_back(*car.clone());
            read_list(cdr, env, v)
        }
        _ => v.push_back(sexp.clone()),
    }
}

mod builtin {
    use std::f64::EPSILON;
    use std::collections::VecDeque;
    use std::iter;
    use std::rc::Rc;
    use itertools::Itertools;

    use super::super::LispError;
    use super::super::sexp::Sexp;
    use super::{eval_call_nonspecial, eval_sexp, read_list, Env, Value};

    #[cfg_attr(feature = "cargo-clippy", allow(needless_pass_by_value))]
    pub fn if_(args: VecDeque<Sexp>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 3 {
            Err(LispError::BadArity("if"))
        } else {
            let cond = eval_sexp(&args[0], env)?;
            match *cond {
                Value::Bool(b) if b => eval_sexp(&args[1], env),
                Value::Bool(b) if !b => eval_sexp(&args[2], env),
                _ => Err(LispError::Runtime("if took non-boolean condition")),
            }
        }
    }

    pub fn lambda(mut args: VecDeque<Sexp>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("lambda"))
        } else {
            let mut params_sexp = VecDeque::new();
            read_list(&args[0], env, &mut params_sexp);
            let mut params = Vec::with_capacity(params_sexp.len());
            for sexp in params_sexp {
                match sexp {
                    Sexp::Ident(s) => params.push(s),
                    Sexp::Null => break,
                    x => return Err(LispError::InvalidLambdaParameter(x)),
                }
            }
            let body = args.pop_back().unwrap();
            Ok(Rc::new(Value::Lambda(params, Rc::new(body))))
        }
    }

    pub fn and(args: VecDeque<Sexp>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        args.into_iter()
            .fold(Ok(Rc::new(Value::Bool(true))), |acc, x| {
                acc.and_then(|acc| {
                    let x = eval_sexp(&x, env)?;
                    match (&*acc, &*x) {
                        (&Value::Bool(false), _) => Ok(Rc::new(Value::Bool(false))),
                        (&Value::Bool(true), &Value::Bool(b)) => Ok(Rc::new(Value::Bool(b))),
                        _ => Err(LispError::ExpectedBool(
                            "and",
                            Rc::try_unwrap(x).unwrap_or_else(|x| (*x).clone()),
                        )),
                    }
                })
            })
    }

    pub fn or(args: VecDeque<Sexp>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        args.into_iter()
            .fold(Ok(Rc::new(Value::Bool(false))), |acc, x| {
                acc.and_then(|acc| {
                    let x = eval_sexp(&x, env)?;
                    match (&*acc, &*x) {
                        (&Value::Bool(true), _) => Ok(Rc::new(Value::Bool(true))),
                        (&Value::Bool(false), &Value::Bool(b)) => Ok(Rc::new(Value::Bool(b))),
                        _ => Err(LispError::ExpectedBool(
                            "or",
                            Rc::try_unwrap(x).unwrap_or_else(|x| (*x).clone()),
                        )),
                    }
                })
            })
    }

    pub fn add(args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        args.into_iter()
            .fold(Ok(Rc::new(Value::Integer(0))), |acc, x| {
                acc.and_then(|acc| match (&*acc, &*x) {
                    (&Value::Integer(l), &Value::Integer(r)) => Ok(Rc::new(Value::Integer(l + r))),
                    (&Value::Integer(l), &Value::Real(r)) => {
                        Ok(Rc::new(Value::Real(f64::from(l) + r)))
                    }
                    (&Value::Real(l), &Value::Integer(r)) => {
                        Ok(Rc::new(Value::Real(l + f64::from(r))))
                    }
                    (&Value::Real(l), &Value::Real(r)) => Ok(Rc::new(Value::Real(l + r))),
                    _ => Err(LispError::ExpectedNumber(
                        "add",
                        Rc::try_unwrap(x).unwrap_or_else(|x| (*x).clone()),
                    )),
                })
            })
    }

    pub fn sub(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if let Some(init) = args.pop_front() {
            if args.is_empty() {
                // negate
                match *init {
                    Value::Integer(x) => Ok(Rc::new(Value::Integer(-x))),
                    Value::Real(x) => Ok(Rc::new(Value::Real(-x))),
                    _ => Err(LispError::ExpectedNumber(
                        "sub",
                        Rc::try_unwrap(init).unwrap_or_else(|x| (*x).clone()),
                    )),
                }
            } else {
                match *init {
                    Value::Integer(_) | Value::Real(_) => (),
                    _ => {
                        return Err(LispError::ExpectedNumber(
                            "sub",
                            Rc::try_unwrap(init).unwrap_or_else(|x| (*x).clone()),
                        ))
                    }
                };
                args.into_iter().fold(Ok(init), |acc, x| {
                    acc.and_then(|acc| match (&*acc, &*x) {
                        (&Value::Integer(l), &Value::Integer(r)) => {
                            Ok(Rc::new(Value::Integer(l - r)))
                        }
                        (&Value::Integer(l), &Value::Real(r)) => {
                            Ok(Rc::new(Value::Real(f64::from(l) - r)))
                        }
                        (&Value::Real(l), &Value::Integer(r)) => {
                            Ok(Rc::new(Value::Real(l - f64::from(r))))
                        }
                        (&Value::Real(l), &Value::Real(r)) => Ok(Rc::new(Value::Real(l - r))),
                        _ => Err(LispError::ExpectedNumber(
                            "sub",
                            Rc::try_unwrap(x).unwrap_or_else(|x| (*x).clone()),
                        )),
                    })
                })
            }
        } else {
            Ok(Rc::new(Value::Integer(0)))
        }
    }

    pub fn not(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if let Some(v) = args.pop_front() {
            match *v {
                Value::Bool(false) => Ok(Rc::new(Value::Bool(true))),
                _ => Ok(Rc::new(Value::Bool(false))),
            }
        } else {
            Err(LispError::BadArity("not"))
        }
    }

    pub fn mul(args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        args.into_iter()
            .fold(Ok(Rc::new(Value::Integer(1))), |acc, x| {
                acc.and_then(|acc| match (&*acc, &*x) {
                    (&Value::Integer(l), &Value::Integer(r)) => Ok(Rc::new(Value::Integer(l * r))),
                    (&Value::Integer(l), &Value::Real(r)) => {
                        Ok(Rc::new(Value::Real(f64::from(l) * r)))
                    }
                    (&Value::Real(l), &Value::Integer(r)) => {
                        Ok(Rc::new(Value::Real(l * f64::from(r))))
                    }
                    (&Value::Real(l), &Value::Real(r)) => Ok(Rc::new(Value::Real(l * r))),
                    _ => Err(LispError::ExpectedNumber(
                        "mul",
                        Rc::try_unwrap(x).unwrap_or_else(|x| (*x).clone()),
                    )),
                })
            })
    }

    pub fn div(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if let Some(zv) = args.pop_front() {
            match *zv {
                Value::Integer(_) | Value::Real(_) => (),
                _ => {
                    return Err(LispError::ExpectedNumber(
                        "div",
                        Rc::try_unwrap(zv).unwrap_or_else(|x| (*x).clone()),
                    ))
                }
            }
            if args.is_empty() {
                // always return float for 1/z
                match *zv {
                    Value::Integer(z) => Ok(Rc::new(Value::Real(1f64 / f64::from(z)))),
                    Value::Real(z) => Ok(Rc::new(Value::Real(1f64 / z))),
                    _ => unreachable!(),
                }
            } else {
                match *zv {
                    Value::Integer(_) | Value::Real(_) => (),
                    _ => {
                        return Err(LispError::ExpectedNumber(
                            "sub",
                            Rc::try_unwrap(zv).unwrap_or_else(|x| (*x).clone()),
                        ))
                    }
                };
                args.into_iter().fold(Ok(zv), |acc, x| {
                    acc.and_then(|acc| match (&*acc, &*x) {
                        (&Value::Integer(l), &Value::Integer(r)) => {
                            Ok(Rc::new(Value::Integer(l / r)))
                        }
                        (&Value::Integer(l), &Value::Real(r)) => {
                            Ok(Rc::new(Value::Real(f64::from(l) / r)))
                        }
                        (&Value::Real(l), &Value::Integer(r)) => {
                            Ok(Rc::new(Value::Real(l / f64::from(r))))
                        }
                        (&Value::Real(l), &Value::Real(r)) => Ok(Rc::new(Value::Real(l / r))),
                        _ => Err(LispError::ExpectedNumber(
                            "div",
                            Rc::try_unwrap(x).unwrap_or_else(|x| (*x).clone()),
                        )),
                    })
                })
            }
        } else {
            Err(LispError::BadArity("div"))
        }
    }

    pub fn quotient(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("quotient"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (&*n, &*m) {
                (&Value::Integer(l), &Value::Integer(r)) => Ok(Rc::new(Value::Integer(l / r))),
                (&Value::Integer(l), &Value::Real(r)) => {
                    Ok(Rc::new(Value::Real(f64::from(l / r as i32))))
                }
                (&Value::Real(l), &Value::Integer(r)) => {
                    Ok(Rc::new(Value::Real(f64::from(l as i32 / r))))
                }
                (&Value::Real(l), &Value::Real(r)) => {
                    Ok(Rc::new(Value::Real(f64::from(l as i32 / r as i32))))
                }
                (&Value::Integer(_), _) | (&Value::Real(_), _) => Err(LispError::ExpectedNumber(
                    "quotient",
                    Rc::try_unwrap(m).unwrap_or_else(|x| (*x).clone()),
                )),
                _ => Err(LispError::ExpectedNumber(
                    "quotient",
                    Rc::try_unwrap(n).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn remainder(
        mut args: VecDeque<Rc<Value>>,
        _env: &mut Env,
    ) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("remainder"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (&*n, &*m) {
                (&Value::Integer(l), &Value::Integer(r)) => Ok(Rc::new(Value::Integer(l % r))),
                (&Value::Integer(l), &Value::Real(r)) => Ok(Rc::new(Value::Real(f64::from(l) % r))),
                (&Value::Real(l), &Value::Integer(r)) => Ok(Rc::new(Value::Real(l % f64::from(r)))),
                (&Value::Real(l), &Value::Real(r)) => Ok(Rc::new(Value::Real(l % r))),
                (&Value::Integer(_), _) | (&Value::Real(_), _) => Err(LispError::ExpectedNumber(
                    "quotient",
                    Rc::try_unwrap(n).unwrap_or_else(|x| (*x).clone()),
                )),
                _ => Err(LispError::ExpectedNumber(
                    "quotient",
                    Rc::try_unwrap(n).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn modulo(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("modulo"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (&*n, &*m) {
                (&Value::Integer(l), &Value::Integer(r)) => if (l < 0) ^ (r < 0) {
                    Ok(Rc::new(Value::Integer((l % r) + r)))
                } else {
                    Ok(Rc::new(Value::Integer(l % r)))
                },
                (&Value::Integer(l), &Value::Real(r)) => if (l < 0) ^ (r < 0.0) {
                    Ok(Rc::new(Value::Real((f64::from(l) % r) + r)))
                } else {
                    Ok(Rc::new(Value::Real(f64::from(l) % r)))
                },
                (&Value::Real(l), &Value::Integer(r)) => if (l < 0.0) ^ (r < 0) {
                    Ok(Rc::new(Value::Real((l % f64::from(r)) + f64::from(r))))
                } else {
                    Ok(Rc::new(Value::Real(l % f64::from(r))))
                },
                (&Value::Real(l), &Value::Real(r)) => if (l < 0.0) ^ (r < 0.0) {
                    Ok(Rc::new(Value::Real((l % r) + r)))
                } else {
                    Ok(Rc::new(Value::Real(l % r)))
                },
                (&Value::Integer(_), _) | (&Value::Real(_), _) => Err(LispError::ExpectedNumber(
                    "modulo",
                    Rc::try_unwrap(m).unwrap_or_else(|x| (*x).clone()),
                )),
                _ => Err(LispError::ExpectedNumber(
                    "modulo",
                    Rc::try_unwrap(n).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn cons(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("cons"))
        } else {
            let car = args.pop_front().unwrap();
            let cdr = args.pop_front().unwrap();
            Ok(Rc::new(Value::Pair(car, cdr)))
        }
    }

    pub fn car(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if let Some(pair) = args.pop_front() {
            match *pair {
                Value::Pair(ref car, _) => Ok(Rc::clone(car)),
                _ => Err(LispError::Runtime("attempted car of non-pair")),
            }
        } else {
            Err(LispError::BadArity("car"))
        }
    }

    pub fn cdr(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if let Some(pair) = args.pop_front() {
            match *pair {
                Value::Pair(_, ref cdr) => Ok(Rc::clone(cdr)),
                _ => Err(LispError::Runtime("attempted cdr of non-pair")),
            }
        } else {
            Err(LispError::BadArity("cdr"))
        }
    }

    #[cfg_attr(feature = "cargo-clippy", allow(needless_pass_by_value))]
    pub fn eq(args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("eq"))
        } else {
            Ok(Rc::new(Value::Bool(eq_values(&args[0], &args[1]))))
        }
    }

    fn eq_values(left: &Value, right: &Value) -> bool {
        match (left, right) {
            // bool
            (&Value::Bool(l), &Value::Bool(r)) => l == r,
            // numbers
            (&Value::Integer(l), &Value::Integer(r)) => l == r,
            (&Value::Integer(l), &Value::Real(r)) => f64::from(l) == r,
            (&Value::Real(l), &Value::Integer(r)) => l == f64::from(r),
            (&Value::Real(l), &Value::Real(r)) => l == r,
            // char
            (&Value::Char(l), &Value::Char(r)) => l == r,
            // str
            (&Value::Str(ref l), &Value::Str(ref r)) => l == r,
            // pair
            (&Value::Pair(ref l1, ref l2), &Value::Pair(ref r1, ref r2)) => {
                eq_values(&*l1, &*r1) && eq_values(&*l2, &*r2)
            }
            // null
            (&Value::Null, &Value::Null) => true,
            // functions and others
            _ => false,
        }
    }

    pub fn gt(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity(">"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (&*n, &*m) {
                (&Value::Integer(l), &Value::Integer(r)) => Ok(Rc::new(Value::Bool(l > r))),
                (&Value::Integer(l), &Value::Real(r)) => Ok(Rc::new(Value::Bool(f64::from(l) > r))),
                (&Value::Real(l), &Value::Integer(r)) => Ok(Rc::new(Value::Bool(l > f64::from(r)))),
                (&Value::Real(l), &Value::Real(r)) => Ok(Rc::new(Value::Bool(l > r))),
                (&Value::Integer(_), _) | (&Value::Real(_), _) => Err(LispError::ExpectedNumber(
                    ">",
                    Rc::try_unwrap(m).unwrap_or_else(|x| (*x).clone()),
                )),
                _ => Err(LispError::ExpectedNumber(
                    ">",
                    Rc::try_unwrap(n).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn gte(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity(">="))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (&*n, &*m) {
                (&Value::Integer(l), &Value::Integer(r)) => Ok(Rc::new(Value::Bool(l >= r))),
                (&Value::Integer(l), &Value::Real(r)) => {
                    Ok(Rc::new(Value::Bool(f64::from(l) >= r)))
                }
                (&Value::Real(l), &Value::Integer(r)) => {
                    Ok(Rc::new(Value::Bool(l >= f64::from(r))))
                }
                (&Value::Real(l), &Value::Real(r)) => Ok(Rc::new(Value::Bool(l >= r))),
                (&Value::Integer(_), _) | (&Value::Real(_), _) => Err(LispError::ExpectedNumber(
                    ">=",
                    Rc::try_unwrap(m).unwrap_or_else(|x| (*x).clone()),
                )),
                _ => Err(LispError::ExpectedNumber(
                    ">=",
                    Rc::try_unwrap(n).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn lt(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("<"))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (&*n, &*m) {
                (&Value::Integer(l), &Value::Integer(r)) => Ok(Rc::new(Value::Bool(l < r))),
                (&Value::Integer(l), &Value::Real(r)) => Ok(Rc::new(Value::Bool(f64::from(l) < r))),
                (&Value::Real(l), &Value::Integer(r)) => Ok(Rc::new(Value::Bool(l < f64::from(r)))),
                (&Value::Real(l), &Value::Real(r)) => Ok(Rc::new(Value::Bool(l < r))),
                (&Value::Integer(_), _) | (&Value::Real(_), _) => Err(LispError::ExpectedNumber(
                    "<",
                    Rc::try_unwrap(m).unwrap_or_else(|x| (*x).clone()),
                )),
                _ => Err(LispError::ExpectedNumber(
                    "<",
                    Rc::try_unwrap(n).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn lte(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("<="))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (&*n, &*m) {
                (&Value::Integer(l), &Value::Integer(r)) => Ok(Rc::new(Value::Bool(l <= r))),
                (&Value::Integer(l), &Value::Real(r)) => {
                    Ok(Rc::new(Value::Bool(f64::from(l) <= r)))
                }
                (&Value::Real(l), &Value::Integer(r)) => {
                    Ok(Rc::new(Value::Bool(l <= f64::from(r))))
                }
                (&Value::Real(l), &Value::Real(r)) => Ok(Rc::new(Value::Bool(l <= r))),
                (&Value::Integer(_), _) | (&Value::Real(_), _) => Err(LispError::ExpectedNumber(
                    "<=",
                    Rc::try_unwrap(m).unwrap_or_else(|x| (*x).clone()),
                )),
                _ => Err(LispError::ExpectedNumber(
                    "<=",
                    Rc::try_unwrap(n).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn neq(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("!="))
        } else {
            let n = args.pop_front().unwrap();
            let m = args.pop_front().unwrap();
            match (&*n, &*m) {
                (&Value::Integer(l), &Value::Integer(r)) => Ok(Rc::new(Value::Bool(l != r))),
                (&Value::Integer(l), &Value::Real(r)) => {
                    Ok(Rc::new(Value::Bool((f64::from(l) - r).abs() > EPSILON)))
                }
                (&Value::Real(l), &Value::Integer(r)) => {
                    Ok(Rc::new(Value::Bool((l - f64::from(r)).abs() > EPSILON)))
                }
                (&Value::Real(l), &Value::Real(r)) => {
                    Ok(Rc::new(Value::Bool((l - r).abs() > EPSILON)))
                }
                (&Value::Integer(_), _) | (&Value::Real(_), _) => Err(LispError::ExpectedNumber(
                    "!=",
                    Rc::try_unwrap(m).unwrap_or_else(|x| (*x).clone()),
                )),
                _ => Err(LispError::ExpectedNumber(
                    "!=",
                    Rc::try_unwrap(n).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn is_pair(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if let Some(x) = args.pop_front() {
            match *x {
                Value::Pair(_, _) => Ok(Rc::new(Value::Bool(true))),
                _ => Ok(Rc::new(Value::Bool(false))),
            }
        } else {
            Err(LispError::BadArity("pair?"))
        }
    }

    pub fn is_null(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if let Some(x) = args.pop_front() {
            match *x {
                Value::Null => Ok(Rc::new(Value::Bool(true))),
                _ => Ok(Rc::new(Value::Bool(false))),
            }
        } else {
            Err(LispError::BadArity("null?"))
        }
    }

    pub fn identity(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        args.pop_front().ok_or(LispError::BadArity("identity"))
    }

    pub fn list(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if let Some(car) = args.pop_back() {
            let mut l = Value::Pair(car, Rc::new(Value::Null));
            for item in args.into_iter().rev() {
                let cdr = Rc::new(l);
                l = Value::Pair(item, cdr);
            }
            Ok(Rc::new(l))
        } else {
            Ok(Rc::new(Value::Null))
        }
    }

    pub fn list_ref(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("list-ref"))
        } else {
            let lst = args.pop_front().unwrap();
            let idx = args.pop_front().unwrap();
            match (&*lst, &*idx) {
                (&Value::Pair(_, _), &Value::Integer(i)) => if i >= 0 {
                    let lst = list_of_pair(lst);
                    lst.into_iter()
                        .nth(i as usize)
                        .ok_or(LispError::Runtime("list-ref index out of bound"))
                } else {
                    Err(LispError::Runtime("list-ref expected nonnegative integer"))
                },
                _ => Err(LispError::Runtime("list-ref expected pair and integer")),
            }
        }
    }

    fn list_of_pair(mut v: Rc<Value>) -> Vec<Rc<Value>> {
        let mut lst = Vec::new();
        while let Value::Pair(ref car, ref cdr) = *v.clone() {
            lst.push(Rc::clone(car));
            v = Rc::clone(cdr);
        }
        lst.push(v);
        lst
    }

    pub fn append(mut args: VecDeque<Rc<Value>>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        match args.len() {
            0 => Ok(Rc::new(Value::Null)),
            1 => Ok(args.pop_front().unwrap()),
            _ => {
                let mut vs = VecDeque::new();
                for arg in args {
                    match *arg {
                        Value::Null => (),
                        Value::Pair(_, _) => {
                            let mut lst: VecDeque<_> = list_of_pair(arg).into();
                            lst.pop_back(); // no null
                            vs.append(&mut lst);
                        }
                        _ => {
                            return Err(LispError::ExpectedList(
                                "append",
                                Rc::try_unwrap(arg).unwrap_or_else(|x| (*x).clone()),
                            ))
                        }
                    }
                }
                list(vs, env)
            }
        }
    }

    pub fn appendmap(mut args: VecDeque<Rc<Value>>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("append-map"))
        } else {
            let f = args.pop_front().unwrap();
            let xs = args.pop_front().unwrap();
            match *xs {
                Value::Pair(_, _) => {
                    let mut lst = list_of_pair(xs);
                    lst.pop(); // no null
                    let items = lst.into_iter()
                        .zip(iter::repeat(f))
                        .map(|(v, f)| eval_call_nonspecial(&f, vec![v].into(), env))
                        .collect::<Result<VecDeque<_>, LispError>>()?;
                    let mut vs = VecDeque::new();
                    for item in items {
                        match *item {
                            Value::Null => (),
                            Value::Pair(_, _) => {
                                let mut lst: VecDeque<_> = list_of_pair(item).into();
                                lst.pop_back(); // no null
                                vs.append(&mut lst);
                            }
                            _ => {
                                return Err(LispError::ExpectedList(
                                    "append-map",
                                    Rc::try_unwrap(item).unwrap_or_else(|x| (*x).clone()),
                                ))
                            }
                        }
                    }
                    list(vs, env)
                }
                Value::Null => Ok(Rc::new(Value::Null)),
                _ => Err(LispError::ExpectedList(
                    "append-map",
                    Rc::try_unwrap(xs).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn map(mut args: VecDeque<Rc<Value>>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("map"))
        } else {
            let f = args.pop_front().unwrap();
            let xs = args.pop_front().unwrap();
            match *xs {
                Value::Pair(_, _) => {
                    let mut lst = list_of_pair(xs);
                    lst.pop(); // no null
                    let items = lst.into_iter()
                        .zip(iter::repeat(f))
                        .map(|(v, f)| eval_call_nonspecial(&f, vec![v].into(), env))
                        .collect::<Result<VecDeque<_>, LispError>>()?;
                    list(items, env)
                }
                Value::Null => Ok(Rc::new(Value::Null)),
                _ => Err(LispError::ExpectedList(
                    "map",
                    Rc::try_unwrap(xs).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn filter(mut args: VecDeque<Rc<Value>>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("filter"))
        } else {
            let f = args.pop_front().unwrap();
            let xs = args.pop_front().unwrap();
            match *xs {
                Value::Pair(_, _) => {
                    let mut lst = list_of_pair(xs);
                    lst.pop(); // no null
                    let evaluated = lst.into_iter()
                        .zip(iter::repeat(f))
                        .map(|(v, f)| {
                            eval_call_nonspecial(&f, vec![Rc::clone(&v)].into(), env)
                                .map(|x| (x, v))
                        })
                        .collect::<Result<Vec<_>, LispError>>()?; // collect to ensure all Ok
                    let items = evaluated
                        .into_iter()
                        .filter_map(|(x, v)| match *x {
                            Value::Bool(true) => Some(v),
                            _ => None,
                        })
                        .collect();
                    list(items, env)
                }
                Value::Null => Ok(Rc::new(Value::Null)),
                _ => Err(LispError::ExpectedList(
                    "map",
                    Rc::try_unwrap(xs).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn foldl(mut args: VecDeque<Rc<Value>>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 3 {
            Err(LispError::BadArity("foldl"))
        } else {
            let f = args.pop_front().unwrap();
            let init = args.pop_front().unwrap();
            let lst = args.pop_front().unwrap();
            match *lst {
                Value::Pair(_, _) => {
                    let mut cur = init;
                    let mut lst = list_of_pair(lst);
                    lst.pop(); // no null
                    for x in lst {
                        cur = eval_call_nonspecial(&f, vec![x, cur].into(), env)?
                    }
                    Ok(cur)
                }
                Value::Null => Ok(init),
                _ => Err(LispError::ExpectedList(
                    "foldl",
                    Rc::try_unwrap(lst).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn count(mut args: VecDeque<Rc<Value>>, env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("count"))
        } else {
            let f = args.pop_front().unwrap();
            let lst = args.pop_front().unwrap();
            match *lst {
                Value::Pair(_, _) => {
                    let mut count = 0;
                    let mut lst = list_of_pair(lst);
                    lst.pop(); // no null
                    for x in lst {
                        let fx = eval_call_nonspecial(&f, vec![x].into(), env)?;
                        match *fx {
                            Value::Bool(true) => count += 1,
                            Value::Bool(false) => (),
                            _ => {
                                return Err(LispError::ExpectedBool(
                                    "count",
                                    Rc::try_unwrap(fx).unwrap_or_else(|x| (*x).clone()),
                                ))
                            }
                        }
                    }
                    Ok(Rc::new(Value::Integer(count)))
                }
                Value::Null => Ok(Rc::new(Value::Integer(0))),
                _ => Err(LispError::ExpectedList(
                    "count",
                    Rc::try_unwrap(lst).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        }
    }

    pub fn member(mut args: VecDeque<Rc<Value>>, _env: &mut Env) -> Result<Rc<Value>, LispError> {
        if args.len() != 2 {
            Err(LispError::BadArity("member"))
        } else {
            let needle = args.pop_front().unwrap();
            let mut haystack = args.pop_front().unwrap();
            while let Value::Pair(ref car, ref cdr) = *haystack.clone() {
                if eq_values(&needle, car) {
                    return Ok(Rc::new(Value::Pair(Rc::clone(car), Rc::clone(cdr))));
                }
                haystack = Rc::clone(cdr);
            }
            Ok(Rc::new(Value::Bool(false)))
        }
    }

    pub fn string_length(
        mut args: VecDeque<Rc<Value>>,
        _env: &mut Env,
    ) -> Result<Rc<Value>, LispError> {
        if let Some(s) = args.pop_front() {
            match *s {
                Value::Str(_) => match *s {
                    Value::Str(ref s) => Ok(Rc::new(Value::Integer(s.chars().count() as i32))),
                    _ => unreachable!(),
                },
                _ => Err(LispError::ExpectedStr(
                    "string-length",
                    Rc::try_unwrap(s).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        } else {
            Err(LispError::BadArity("string-length"))
        }
    }

    pub fn string_downcase(
        mut args: VecDeque<Rc<Value>>,
        _env: &mut Env,
    ) -> Result<Rc<Value>, LispError> {
        if let Some(s) = args.pop_front() {
            match *s {
                Value::Str(_) => match *s {
                    Value::Str(ref s) => Ok(Rc::new(Value::Str(s.to_lowercase()))),
                    _ => unreachable!(),
                },
                _ => Err(LispError::ExpectedStr(
                    "string-downcase",
                    Rc::try_unwrap(s).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        } else {
            Err(LispError::BadArity("string-downcase"))
        }
    }

    pub fn string_upcase(
        mut args: VecDeque<Rc<Value>>,
        _env: &mut Env,
    ) -> Result<Rc<Value>, LispError> {
        if let Some(s) = args.pop_front() {
            match *s {
                Value::Str(_) => match *s {
                    Value::Str(ref s) => Ok(Rc::new(Value::Str(s.to_uppercase()))),
                    _ => unreachable!(),
                },
                _ => Err(LispError::ExpectedStr(
                    "stirng-upcase",
                    Rc::try_unwrap(s).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        } else {
            Err(LispError::BadArity("string-upcase"))
        }
    }

    pub fn string_append(
        args: VecDeque<Rc<Value>>,
        _env: &mut Env,
    ) -> Result<Rc<Value>, LispError> {
        let strs = args.into_iter()
            .map(|v| match *v {
                Value::Str(_) => match *v {
                    Value::Str(ref s) => Ok(s.clone()),
                    _ => unreachable!(),
                },
                _ => Err(LispError::ExpectedStr(
                    "string-append",
                    Rc::try_unwrap(v).unwrap_or_else(|x| (*x).clone()),
                )),
            })
            .collect::<Result<Vec<_>, LispError>>()?;
        Ok(Rc::new(Value::Str(strs.into_iter().join(""))))
    }

    pub fn substring(
        mut args: VecDeque<Rc<Value>>,
        _env: &mut Env,
    ) -> Result<Rc<Value>, LispError> {
        if !(args.len() == 2 || args.len() == 3) {
            Err(LispError::BadArity("substring"))
        } else {
            let sv = args.pop_front().unwrap();
            let startv = args.pop_front().unwrap();
            let endo = args.pop_front();
            if let Value::Str(_) = *sv {
                match *sv {
                    Value::Str(ref s) => match (&*startv, endo) {
                        (&Value::Integer(start), _) if start < 0 => {
                            Err(LispError::Runtime("list-ref expected nonnegative integer"))
                        }
                        (&Value::Integer(start), None) => Ok(Rc::new(Value::Str(
                            s.chars().skip(start as usize).collect(),
                        ))),
                        (&Value::Integer(start), Some(endv)) => match *endv {
                            Value::Integer(end) if end < start => {
                                Err(LispError::Runtime("list-ref end was less that start"))
                            }
                            Value::Integer(end) => Ok(Rc::new(Value::Str(
                                s.chars()
                                    .skip(start as usize)
                                    .take((end - start) as usize)
                                    .collect(),
                            ))),
                            _ => Err(LispError::ExpectedInteger(
                                "substring",
                                Rc::try_unwrap(endv).unwrap_or_else(|x| (*x).clone()),
                            )),
                        },
                        _ => Err(LispError::ExpectedInteger(
                            "substring",
                            Rc::try_unwrap(startv).unwrap_or_else(|x| (*x).clone()),
                        )),
                    },
                    _ => unreachable!(),
                }
            } else {
                Err(LispError::ExpectedStr(
                    "substring",
                    Rc::try_unwrap(sv).unwrap_or_else(|x| (*x).clone()),
                ))
            }
        }
    }

    pub fn string_split(
        mut args: VecDeque<Rc<Value>>,
        env: &mut Env,
    ) -> Result<Rc<Value>, LispError> {
        if !(args.len() == 1 || args.len() == 2) {
            Err(LispError::BadArity("substring"))
        } else {
            let sv = args.pop_front().unwrap();
            let sepo = args.pop_front();
            if let Value::Str(_) = *sv {
                match *sv {
                    Value::Str(ref s) => match sepo {
                        None => {
                            let vals = s.split_whitespace()
                                .map(|s| Rc::new(Value::Str(String::from(s))))
                                .collect();
                            list(vals, env)
                        }
                        Some(sepv) => match *sepv {
                            Value::Str(_) => match *sepv {
                                Value::Str(ref sep) => {
                                    let vals = s.split(sep)
                                        .map(|s| Rc::new(Value::Str(String::from(s))))
                                        .collect();
                                    list(vals, env)
                                }
                                _ => unreachable!(),
                            },
                            _ => Err(LispError::ExpectedStr(
                                "string-split",
                                Rc::try_unwrap(sepv).unwrap_or_else(|x| (*x).clone()),
                            )),
                        },
                    },
                    _ => unreachable!(),
                }
            } else {
                Err(LispError::ExpectedStr(
                    "string-split",
                    Rc::try_unwrap(sv).unwrap_or_else(|x| (*x).clone()),
                ))
            }
        }
    }

    pub fn string_join(
        mut args: VecDeque<Rc<Value>>,
        _env: &mut Env,
    ) -> Result<Rc<Value>, LispError> {
        if let Some(xs) = args.pop_front() {
            match *xs {
                Value::Null => Ok(Rc::new(Value::Str(String::new()))),
                Value::Pair(_, _) => {
                    let mut lst = list_of_pair(Rc::clone(&xs));
                    lst.pop(); // no null
                    let strs = lst.into_iter()
                        .map(|v| match *v {
                            Value::Str(_) => match *v {
                                Value::Str(ref s) => Ok(s.clone()),
                                _ => unreachable!(),
                            },
                            _ => Err(LispError::ExpectedStr(
                                "string-join",
                                Rc::try_unwrap(v).unwrap_or_else(|x| (*x).clone()),
                            )),
                        })
                        .collect::<Result<Vec<String>, LispError>>()?;
                    match args.pop_front() {
                        None => Ok(Rc::new(Value::Str(strs.into_iter().join("")))),
                        Some(sepv) => match *sepv {
                            Value::Str(_) => match *sepv {
                                Value::Str(ref sep) => {
                                    Ok(Rc::new(Value::Str(strs.into_iter().join(sep))))
                                }
                                _ => unreachable!(),
                            },
                            _ => Err(LispError::ExpectedStr(
                                "string-join",
                                Rc::try_unwrap(sepv).unwrap_or_else(|x| (*x).clone()),
                            )),
                        },
                    }
                }
                _ => Err(LispError::ExpectedList(
                    "string-join",
                    Rc::try_unwrap(xs).unwrap_or_else(|x| (*x).clone()),
                )),
            }
        } else {
            Err(LispError::BadArity("string-join"))
        }
    }
}
