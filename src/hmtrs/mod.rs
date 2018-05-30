//! (representation)
//! A [Hindley-Milner][1] [Term Rewriting System][0] (HMTRS).
//!
//! [0]: https://wikipedia.org/wiki/Hindley–Milner_type_system
//!      "Wikipedia - Hindley-Milner Type System"
//! [1]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//!      "Wikipedia - Term Rewriting Systems"
use super::Task;
use polytype::TypeSchema;
use rand::random;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::{INFINITY, NEG_INFINITY};
use std::iter::once;
use term_rewriting::{Rule, Signature, Term, TRS};

/// A Hindley-Milner Term Rewriting System (HMTRS): a first-order [term rewriting system][0] with a [Hindley-Milner type system][1].
///
/// [0]: https://wikipedia.org/wiki/Hindley–Milner_type_system
///      "Wikipedia - Hindley-Milner Type System"
/// [1]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
///      "Wikipedia - Term Rewriting Systems"
#[derive(Debug, PartialEq, Clone)]
pub struct HMTRS {
    // TODO: may also want to track background knowledge here.
    ops: Vec<TypeSchema>,
    vars: Vec<TypeSchema>,
    rules: Vec<(TypeSchema, TypeSchema, Vec<TypeSchema>)>,
    signature: Signature,
    trs: TRS,
}
impl HMTRS {
    /// the size of the HMTRS (the sum over the size of the rules in the [`TRS`])
    ///
    /// [`TRS`]: ../../term_rewriting/struct.TRS.html
    pub fn size(&self) -> usize {
        self.trs.size()
    }
    pub fn display(&self) -> String {
        self.trs.display(&self.signature)
    }
    // TODO:  typechecking for terms
    // TODO:  typechecking for rules
    // TODO:  typechecking for TRSs
}

pub struct Trace<'a> {
    nodes: Vec<TraceNodeStore>,
    trs: &'a TRS,
    unobserved: BinaryHeap<TraceNode>,
    p_observe: f64,
    max_steps: usize,
    steps: usize,
    max_size: usize,
}
impl<'a> Trace<'a> {
    pub fn new(
        trs: &'a TRS,
        term: &Term,
        p_observe: f64,
        max_steps: usize,
        max_size: usize,
    ) -> Trace<'a> {
        let nodes = vec![TraceNodeStore {
            term: term.clone(),
            state: TraceState::Start,
            log_p: 0.0,
            depth: 0,
            parent: None,
            children: vec![],
        }];
        let unobserved = BinaryHeap::from(vec![TraceNode {
            id: 0,
            score: nodes[0].log_p,
        }]);
        Trace {
            nodes,
            trs,
            unobserved,
            p_observe,
            max_steps,
            max_size,
            steps: 0,
        }
    }
    pub fn new_node(
        &mut self,
        term: Term,
        parent: Option<TraceNode>,
        state: TraceState,
        log_p: f64,
    ) -> TraceNode {
        let node = TraceNodeStore {
            term,
            parent,
            log_p,
            state,
            children: vec![],
            depth: if let Some(x) = parent {
                self.nodes[x.id].depth + 1
            } else {
                0
            },
        };
        self.nodes.push(node);
        TraceNode {
            id: self.nodes.len(),
            score: log_p,
        }
    }
    /// The input node of the Trace
    pub fn root(&self) -> TraceNode {
        TraceNode {
            id: 0,
            score: self.nodes[0].log_p,
        }
    }
    fn node(&self, node: TraceNode) -> &TraceNodeStore {
        &self.nodes[node.id]
    }
    fn mut_node(&mut self, node: TraceNode) -> &mut TraceNodeStore {
        &mut self.nodes[node.id]
    }
    /// All the nodes descending from some Node in the Trace.
    pub fn nodes(&self, node: TraceNode, states: &[TraceState]) -> Vec<TraceNode> {
        let mut all_nodes = self.nodes_helper(node);
        all_nodes.retain(|x| states.contains(&x.state(self)));
        all_nodes
    }
    fn nodes_helper(&self, node: TraceNode) -> Vec<TraceNode> {
        let child_nodes = self.node(node)
            .children
            .iter()
            .flat_map(|&x| self.nodes_helper(x));
        once(node).chain(child_nodes).collect()
    }
    /// Is this node a leaf?
    pub fn is_leaf(&self, node: TraceNode) -> bool {
        self.node(node).children.is_empty()
    }
    pub fn leaves(&self, node: TraceNode, states: &[TraceState]) -> Vec<TraceNode> {
        self.nodes(node, states)
            .into_iter()
            .filter(|&x| self.is_leaf(x))
            .collect()
    }
    /// Return the number of proper descendants of `node` (i.e. excludes `node`).
    pub fn num_descendents(&self, node: TraceNode) -> usize {
        self.nodes(node, &[]).len() - 1
    }
    /// What's the longest series of evaluation steps?
    pub fn depth(&self) -> usize {
        self.leaves(self.root(), &[])
            .iter()
            .map(|x| x.depth(self))
            .max()
            .unwrap_or(0)
    }
    /// How many nodes are in the Trace?
    pub fn size(&self) -> usize {
        self.nodes(self.root(), &[]).len()
    }
    /// How much probability mass has been explored in the Trace?
    pub fn mass(&self) -> f64 {
        // the mass is 1 - the mass in unobserved leaves
        let leaf_mass = self.leaves(self.root(), &[TraceState::Start, TraceState::Unobserved])
            .iter()
            .map(|x| x.log_p(self))
            .chain(once(-INFINITY))
            .collect::<Vec<f64>>();
        1.0 - logsumexp(leaf_mass.as_slice()).exp()
    }
    /// Give one possible outcome in proportion to its probability.
    pub fn sample(&self) -> TraceNode {
        let leaves = self.leaves(self.root(), &[]);
        let ps = leaves.iter().map(|x| x.log_p(self)).collect::<Vec<f64>>();
        weighted_sample(&leaves, ps.as_slice()).unwrap_or_else(|| self.root())
    }
    /// Return the leaf terms of the Trace.
    pub fn leaf_terms(&self, states: &[TraceState]) -> Vec<Term> {
        self.leaves(self.root(), states)
            .iter()
            .map(|x| x.term(self))
            .collect()
    }
    pub fn step(&mut self) -> Option<TraceNode> {
        match self.unobserved.pop() {
            Some(handle) if self.node(handle).term.size() > self.max_size => {
                self.mut_node(handle).state = TraceState::TooBig;
                None
            }
            Some(handle) if self.steps < self.max_steps => {
                match self.trs.rewrite(&self.nodes[handle.id].term) {
                    Some(ref rewrites) if !rewrites.is_empty() => {
                        for term in rewrites {
                            let new_p = self.node(handle).log_p + (1.0 - self.p_observe).ln()
                                - log_n_of(rewrites, 1, 0.0);
                            let unobserved = self.new_node(
                                term.clone(),
                                Some(handle),
                                TraceState::Unobserved,
                                new_p,
                            );
                            self.unobserved.push(TraceNode {
                                id: unobserved.id,
                                score: new_p,
                            });
                            self.mut_node(handle).children.push(unobserved);
                        }
                        self.mut_node(handle).log_p += self.p_observe.ln()
                    }
                    _ => {
                        self.mut_node(handle).state = TraceState::Normal;
                    }
                }
                self.steps += 1;
                Some(handle)
            }
            _ => None,
        }
    }
    /// Run the Trace.
    pub fn run(&mut self) -> &Trace {
        loop {
            if self.step().is_none() {
                return self;
            }
        }
    }
    /// Run the Trace and Return the leaf terms.
    pub fn rewrite(&mut self, states: &[TraceState]) -> Vec<Term> {
        self.run().leaf_terms(states)
    }
    /// With what probability does `self` rewrite to `term`?
    pub fn rewrites_to(&mut self, term: &Term) -> f64 {
        // NOTE: we only use tree equality and don't consider tree edit distance
        self.run();
        let lps = self.leaves(self.root(), &[]) // is this right?
            .iter()
            .filter(|x| Term::alpha(term, &x.term(self)).is_some())
            .map(|x| x.log_p(self))
            .collect::<Vec<f64>>();
        if lps.is_empty() {
            NEG_INFINITY
        } else {
            logsumexp(&lps)
        }
    }
}

/// A single node in a Trace.
#[derive(Debug, Clone)]
struct TraceNodeStore {
    term: Term,
    log_p: f64,
    state: TraceState,
    depth: usize,
    parent: Option<TraceNode>,
    children: Vec<TraceNode>,
}

/// The state of a Node in a Trace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraceState {
    Start,
    Normal,
    Unobserved,
    TooBig,
}

/// The public interface for a TraceNodeStore
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceNode {
    id: usize,
    score: f64,
}
impl TraceNode {
    pub fn state(&self, t: &Trace) -> TraceState {
        t.nodes[self.id].state
    }
    pub fn term(&self, t: &Trace) -> Term {
        t.nodes[self.id].term.clone()
    }
    pub fn log_p(&self, t: &Trace) -> f64 {
        t.nodes[self.id].log_p
    }
    pub fn depth(&self, t: &Trace) -> usize {
        t.nodes[self.id].depth
    }
    pub fn parent(&self, t: &Trace) -> Option<TraceNode> {
        t.nodes[self.id].parent
    }
    pub fn children(&self, t: &Trace) -> Vec<TraceNode> {
        t.nodes[self.id].children.clone()
    }
}
impl Eq for TraceNode {}
impl PartialOrd for TraceNode {
    fn partial_cmp(&self, other: &TraceNode) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for TraceNode {
    fn cmp(&self, other: &TraceNode) -> Ordering {
        if let Some(x) = self.partial_cmp(&other) {
            x
        } else {
            Ordering::Equal
        }
    }
}

pub fn make_task_from_data(
    data: &[Rule],
    tp: TypeSchema,
    p_partial: f64,
    temperature: f64,
    prior_temperature: f64,
    ll_temperature: f64,
) -> Task<HMTRS, (), ()> {
    Task {
        oracle: Box::new(move |h: &HMTRS, _x| {
            posterior(
                h,
                data,
                p_partial,
                temperature,
                prior_temperature,
                ll_temperature,
            ) // TODO: not getting information from true evaluation
        }),
        // TODO: compute type schema from the data
        tp,
        observation: (),
    }
}

pub fn posterior(
    h: &HMTRS,
    data: &[Rule],
    p_partial: f64,
    temperature: f64,
    prior_temperature: f64,
    ll_temperature: f64,
) -> f64 {
    let prior = pseudo_log_prior(h, temperature, prior_temperature);
    if prior == NEG_INFINITY {
        NEG_INFINITY
    } else {
        prior + log_likelihood(h, data, p_partial, temperature, ll_temperature)
    }
}

pub fn pseudo_log_prior(h: &HMTRS, temp: f64, prior_temp: f64) -> f64 {
    let raw_prior = -(h.size() as f64);
    raw_prior / ((temp + 1.0) * prior_temp)
}

pub fn log_likelihood(h: &HMTRS, data: &[Rule], p_partial: f64, temp: f64, ll_temp: f64) -> f64 {
    data.iter()
        .map(|x| single_log_likelihood(h, x, p_partial, temp) / ll_temp)
        .sum()
}

pub fn single_log_likelihood(h: &HMTRS, datum: &Rule, p_partial: f64, temp: f64) -> f64 {
    let p_observe = 0.0;
    let max_steps = 50;
    let max_size = 500;
    let mut trace = Trace::new(&h.trs, &datum.lhs, p_observe, max_steps, max_size);
    trace.run();

    let ll = if let Some(ref rhs) = datum.rhs() {
        trace.rewrites_to(rhs)
    } else {
        NEG_INFINITY
    };

    if ll == NEG_INFINITY {
        (p_partial + temp).ln()
    } else {
        (1.0 - p_partial + temp).ln() + ll
    }
}

pub fn log_n_of<T>(xs: &[T], n: usize, empty: f64) -> f64 {
    if xs.is_empty() {
        empty.ln()
    } else {
        (n as f64).ln() - (xs.len() as f64).ln()
    }
}

pub fn weighted_sample<T: Clone>(xs: &[T], ps: &[f64]) -> Option<T> {
    if xs.len() == ps.len() {
        let cumsum = ps.iter()
            .scan(0.0, |acc, x| {
                *acc += *x;
                Some(*acc)
            })
            .collect::<Vec<f64>>();
        let threshold: f64 = random();
        for (i, &c) in cumsum.iter().enumerate() {
            if c <= threshold {
                return Some(xs[i].clone());
            }
        }
    }
    None
}

pub fn logsumexp(lps: &[f64]) -> f64 {
    if lps.is_empty() {
        return NEG_INFINITY;
    }
    let mut a = NEG_INFINITY;
    for &lp in lps {
        if lp > a {
            a = lp;
        }
    }
    let x = lps.iter().map(|lp| (lp - a).exp()).sum::<f64>().ln();
    a + x
}
