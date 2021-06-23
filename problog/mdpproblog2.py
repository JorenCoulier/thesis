#! /usr/bin/env python

from __future__ import print_function

import time

import sys

from ..program import PrologFile
from ..engine import DefaultEngine
from ..logic import Term
from .. import get_evaluatables, get_evaluatable
from problog.logic import Constant


def main(argv):
    model = PrologFile(argv[0])
    result = mdpproblog(model)
    state_values_pretty, state_decisions_pretty = result
    print('STATE VALUES:')
    print_state_values(state_values_pretty)
    print()
    print('///////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print()
    print('STATE DECISIONS:')
    print_state_decisions(state_decisions_pretty)


def print_state_values(to_print):
    for state, value in to_print:
        print('state: ', end='')
        for s in state:
            print(s, end=' ')

        print('    ', end='')
        print('value: ', end='')
        print(value)

    return None


def print_state_decisions(to_print):
    for state, decisions in to_print:
        print('state: ', end='')
        for s in state:
            print(s, end=' ')

        print('    ', end='')
        print('decisions: ', end='')
        for d in decisions:
            print(d, end=' ')

        print()

    return None


def create_state_variables2(state_variables, name):
    return [encapsulate(i, name) for i in state_variables]


def encapsulate(term, name):
    return Term(name, term)


def get_variable_ordering2(states):
    return [to_next_state2(s) for s in states]


def to_next_state2(state):
    return Term('next', *state.args)


def solve_mdp_factor_single_decision(formula, decisions, current_state_variables, utilities, next_state_variables, constraints):
    counter = 0  # TODO test counter
    time1 = time.perf_counter()
    state_ids, state_names = zip(*current_state_variables)
    state_variable_ordering = get_variable_ordering2(state_names)
    decision_ids, decision_names = zip(*decisions)

    transition_probabilities = dict(((i, j), dict((k, 0) for k in next_state_variables))
                                    for i in range(0, 1 << len(current_state_variables))
                                    for j in range(0, len(decisions)))
    utility_values = dict(((i, j), 0)
                          for i in range(0, 1 << len(current_state_variables))
                          for j in range(0, len(decisions)))

    for i in range(0, 1 << len(current_state_variables)):
        state = num2bits(i, len(current_state_variables))
        evidence_state = dict(zip(state_names, map(int, state)))

        constraints_ok = True
        for c in constraints:
            if not c.check(dict(zip(state_ids, map(int, state)))):
                constraints_ok = False
                break
        if not constraints_ok:
            continue

        for j in range(0, len(decisions)):
            choices = num2bits(1 << j, len(decisions))

            evidence_decisions = dict(zip(decision_names, map(int, choices)))
            evidence = evidence_decisions.copy()
            evidence.update(evidence_state)

            constraints_ok = True
            for c in constraints:
                if not c.check(dict(zip(decision_ids, map(int, choices)))):
                    constraints_ok = False
                    break
            if not constraints_ok:
                continue

            constraints_ok = True
            for c in constraints:
                if not c.check(dict(zip(state_ids + decision_ids, map(int, state + choices)))):
                    constraints_ok = False
                    break
            if not constraints_ok:
                continue

            counter += 1  # TODO test counter increment
            get_transition_probabilities(formula, evidence_state, evidence_decisions, utilities, next_state_variables,
                                         transition_probabilities, utility_values, i, j)

    time2 = time.perf_counter()
    prob_time = time2 - time1
    print('Probability calculation time: ', prob_time)
    print('Number of evaluations: ', counter)
    state_values, state_decisions = value_iteration_factor(1 << len(current_state_variables), len(decisions),
                                                           transition_probabilities, utility_values,
                                                           state_variable_ordering)
    state_values_pretty = []
    state_decisions_pretty = []
    for i in range(0, 1 << len(current_state_variables)):
        state_value = state_values[i]
        state_bits = num2bits(i, len(current_state_variables))
        state_pretty = zip(state_names, state_bits)
        state_values_pretty.append((state_pretty, state_value))

        state_decision = state_decisions[i]
        decision_pretty = decision_names[len(decision_names) - 1 - state_decision]
        state_pretty2 = zip(state_names, state_bits)
        state_decisions_pretty.append((state_pretty2, [decision_pretty]))

    return state_values_pretty, state_decisions_pretty


def solve_mdp_factor(formula, decisions, current_state_variables, utilities, next_state_variables, constraints):
    counter = 0  # TODO test counter
    time1 = time.perf_counter()
    state_ids, state_names = zip(*current_state_variables)
    state_variable_ordering = get_variable_ordering2(state_names)
    decision_ids, decision_names = zip(*decisions)

    transition_probabilities = dict(((i, j), dict((k, 0) for k in next_state_variables))
                                    for i in range(0, 1 << len(current_state_variables))
                                    for j in range(0, 1 << len(decisions)))
    utility_values = dict(((i, j), 0)
                          for i in range(0, 1 << len(current_state_variables))
                          for j in range(0, 1 << len(decisions)))

    for i in range(0, 1 << len(current_state_variables)):
        state = num2bits(i, len(current_state_variables))
        evidence_state = dict(zip(state_names, map(int, state)))

        constraints_ok = True
        for c in constraints:
            if not c.check(dict(zip(state_ids, map(int, state)))):
                constraints_ok = False
                break
        if not constraints_ok:
            continue

        for j in range(0, 1 << len(decisions)):
            choices = num2bits(j, len(decisions))

            evidence_decisions = dict(zip(decision_names, map(int, choices)))
            evidence = evidence_decisions.copy()
            evidence.update(evidence_state)

            constraints_ok = True
            for c in constraints:
                if not c.check(dict(zip(decision_ids, map(int, choices)))):
                    constraints_ok = False
                    break
            if not constraints_ok:
                continue

            constraints_ok = True
            for c in constraints:
                if not c.check(dict(zip(state_ids + decision_ids, map(int, state + choices)))):
                    constraints_ok = False
                    break
            if not constraints_ok:
                continue

            counter += 1  # TODO test counter increment
            get_transition_probabilities(formula, evidence_state, evidence_decisions, utilities, next_state_variables,
                                         transition_probabilities, utility_values, i, j)

    time2 = time.perf_counter()
    prob_time = time2 - time1
    print('Probability calculation time: ', prob_time)
    print('Number of evaluations: ', counter)
    state_values, state_decisions = value_iteration_factor(1 << len(current_state_variables), 1 << len(decisions),
                                                           transition_probabilities, utility_values,
                                                           state_variable_ordering)
    state_values_pretty = []
    state_decisions_pretty = []
    for i in range(0, 1 << len(current_state_variables)):
        state_value = state_values[i]
        state_bits = num2bits(i, len(current_state_variables))
        state_pretty = zip(state_names, state_bits)
        state_values_pretty.append((state_pretty, state_value))

        state_decision = state_decisions[i]
        decision_bits = num2bits(state_decision, len(decisions))
        decision_pretty = zip(decision_names, decision_bits)
        state_pretty2 = zip(state_names, state_bits)
        state_decisions_pretty.append((state_pretty2, decision_pretty))

    return state_values_pretty, state_decisions_pretty


def solve_mdp_both_single_decision(formula, decisions, current_state_variables_factor, current_state_variables_discrete,
                                   utilities, next_state_variables_factor, next_state_variables_discrete, constraints):
    """

    :param formula:
    :param decisions:
    :param current_state_variables_factor:
    :param current_state_variables_discrete:
    :param utilities:
    :param next_state_variables_factor:
    :param next_state_variables_discrete:
    :param constraints:
    :return:

    extra comments:
        The transition_probabilities dictionary returns a dictionary for a tuple
        (current factored state, current discrete state, decision).
        The returned disctionary contains the transition probabilities for each next state variable
        (both factored and discrete).

        The utility_values dictionary returns a number for a tuple
        (current factored state, current discrete state, decision).
        This number is the value of the state and decision that correspond with the tuple.
    """
    counter = 0  # TODO test counter
    time1 = time.perf_counter()
    state_ids_factor, state_names_factor = zip(*current_state_variables_factor)
    state_ids_discrete, state_names_discrete = zip(*current_state_variables_discrete)
    state_variable_ordering_factor = get_variable_ordering2(state_names_factor)
    state_variable_ordering_discrete = get_variable_ordering2(state_names_discrete)
    decision_ids, decision_names = zip(*decisions)
    next_state_variables_all = next_state_variables_factor + next_state_variables_discrete

    transition_probabilities = dict(((i1, i2, j), dict((k, 0) for k in next_state_variables_all))
                                    for i1 in range(0, 1 << len(current_state_variables_factor))
                                    for i2 in range(0, len(current_state_variables_discrete))
                                    for j in range(0, len(decisions)))

    utility_values = dict(((i1, i2, j), 0)
                          for i1 in range(0, 1 << len(current_state_variables_factor))
                          for i2 in range(0, len(current_state_variables_discrete))
                          for j in range(0, len(decisions)))

    for i1 in range(0, 1 << len(current_state_variables_factor)):
        state_factor = num2bits(i1, len(current_state_variables_factor))
        evidence_state_factor = dict(zip(state_names_factor, map(int, state_factor)))

        constraints_ok = True
        for c in constraints:
            if not c.check(dict(zip(state_ids_factor, map(int, state_factor)))):
                constraints_ok = False
                break
        if not constraints_ok:
            continue

        for i2 in range(0, len(current_state_variables_discrete)):
            state_discrete = num2bits(1 << i2, len(current_state_variables_discrete))
            evidence_state_discrete = dict(zip(state_names_discrete, map(int, state_discrete)))

            constraints_ok = True
            for c in constraints:
                if not c.check(dict(zip(state_ids_discrete, map(int, state_discrete)))):
                    constraints_ok = False
                    break
            if not constraints_ok:
                continue

            for j in range(0, len(decisions)):
                choices = num2bits(1 << j, len(decisions))

                evidence_decisions = dict(zip(decision_names, map(int, choices)))
                evidence = evidence_decisions.copy()
                evidence.update(evidence_state_factor)
                evidence.update(evidence_state_discrete)

                constraints_ok = True
                for c in constraints:
                    if not c.check(dict(zip(decision_ids, map(int, choices)))):
                        constraints_ok = False
                        break
                if not constraints_ok:
                    continue

                constraints_ok = True
                for c in constraints:
                    if not c.check(dict(zip(state_ids_factor + state_ids_discrete + decision_ids,
                                            map(int, state_factor + state_discrete + choices)))):
                        constraints_ok = False
                        break
                if not constraints_ok:
                    continue

                counter += 1  # TODO test counter increment
                get_transition_probabilities_both(formula, evidence, utilities, next_state_variables_all,
                                                  transition_probabilities, utility_values, i1, i2, j)

    time2 = time.perf_counter()
    prob_time = time2 - time1
    print('Probability calculation time: ', prob_time)
    print('Number of evaluations: ', counter)
    state_values, state_decisions = value_iteration_both(1 << len(current_state_variables_factor),
                                                         len(current_state_variables_discrete), len(decisions),
                                                         transition_probabilities, utility_values,
                                                         state_variable_ordering_factor, state_variable_ordering_discrete)
    state_values_pretty = []
    state_decisions_pretty = []
    for i1 in range(0, 1 << len(current_state_variables_factor)):
        for i2 in range(0, len(current_state_variables_discrete)):
            state_value = state_values[i1, i2]
            state_bits_factor = num2bits(i1, len(current_state_variables_factor))
            state_pretty_factor = list(zip(state_names_factor, state_bits_factor))
            state_pretty_discrete = [state_names_discrete[len(state_names_discrete) - 1 - i2]]
            state_pretty = state_pretty_factor + state_pretty_discrete
            state_values_pretty.append((state_pretty, state_value))

            state_decision = state_decisions[i1, i2]
            # decision_bits = num2bits(state_decision, len(decisions))
            # decision_pretty = zip(decision_names, decision_bits)
            decision_pretty = decision_names[len(decision_names) - 1 - state_decision]
            # state_pretty2 = zip(state_names, state_bits)
            state_decisions_pretty.append((state_pretty, [decision_pretty]))

    return state_values_pretty, state_decisions_pretty


# TODO testing multiple ddnnfs instead of one for all queries
def solve_mdp_both_test(formula_utilities, formula_next, decisions, current_state_variables_factor, current_state_variables_discrete,
                        utilities, next_state_variables_factor, next_state_variables_discrete, constraints):

    counter = 0  # TODO test counter
    time1 = time.perf_counter()
    state_ids_factor, state_names_factor = zip(*current_state_variables_factor)
    state_ids_discrete, state_names_discrete = zip(*current_state_variables_discrete)
    state_variable_ordering_factor = get_variable_ordering2(state_names_factor)
    state_variable_ordering_discrete = get_variable_ordering2(state_names_discrete)
    decision_ids, decision_names = zip(*decisions)
    next_state_variables_all = next_state_variables_factor + next_state_variables_discrete

    transition_probabilities = dict(((i1, i2, j), dict((k, 0) for k in next_state_variables_all))
                                    for i1 in range(0, 1 << len(current_state_variables_factor))
                                    for i2 in range(0, len(current_state_variables_discrete))
                                    for j in range(0, len(decisions)))

    utility_values = dict(((i1, i2, j), 0)
                          for i1 in range(0, 1 << len(current_state_variables_factor))
                          for i2 in range(0, len(current_state_variables_discrete))
                          for j in range(0, len(decisions)))

    for i1 in range(0, 1 << len(current_state_variables_factor)):
        state_factor = num2bits(i1, len(current_state_variables_factor))
        evidence_state_factor = dict(zip(state_names_factor, map(int, state_factor)))

        constraints_ok = True
        for c in constraints:
            if not c.check(dict(zip(state_ids_factor, map(int, state_factor)))):
                constraints_ok = False
                break
        if not constraints_ok:
            continue

        for i2 in range(0, len(current_state_variables_discrete)):
            state_discrete = num2bits(1 << i2, len(current_state_variables_discrete))
            evidence_state_discrete = dict(zip(state_names_discrete, map(int, state_discrete)))

            constraints_ok = True
            for c in constraints:
                if not c.check(dict(zip(state_ids_discrete, map(int, state_discrete)))):
                    constraints_ok = False
                    break
            if not constraints_ok:
                continue

            for j in range(0, len(decisions)):
                choices = num2bits(1 << j, len(decisions))

                evidence_decisions = dict(zip(decision_names, map(int, choices)))
                evidence = evidence_decisions.copy()
                evidence.update(evidence_state_factor)
                evidence.update(evidence_state_discrete)

                constraints_ok = True
                for c in constraints:
                    if not c.check(dict(zip(decision_ids, map(int, choices)))):
                        constraints_ok = False
                        break
                if not constraints_ok:
                    continue

                constraints_ok = True
                for c in constraints:
                    if not c.check(dict(zip(state_ids_factor + state_ids_discrete + decision_ids,
                                            map(int, state_factor + state_discrete + choices)))):
                        constraints_ok = False
                        break
                if not constraints_ok:
                    continue

                counter += 1  # TODO test counter increment
                get_transition_probabilities_both(formula_utilities, evidence, utilities, next_state_variables_all,
                                                  transition_probabilities, utility_values, i1, i2, j)
                get_transition_probabilities_both(formula_next, evidence, utilities, next_state_variables_all,
                                                  transition_probabilities, utility_values, i1, i2, j)

    time2 = time.perf_counter()
    prob_time = time2 - time1
    print('Probability calculation time: ', prob_time)
    print('Number of evaluations: ', counter)
    state_values, state_decisions = value_iteration_both(1 << len(current_state_variables_factor),
                                                         len(current_state_variables_discrete), len(decisions),
                                                         transition_probabilities, utility_values,
                                                         state_variable_ordering_factor, state_variable_ordering_discrete)
    state_values_pretty = []
    state_decisions_pretty = []
    for i1 in range(0, 1 << len(current_state_variables_factor)):
        for i2 in range(0, len(current_state_variables_discrete)):
            state_value = state_values[i1, i2]
            state_bits_factor = num2bits(i1, len(current_state_variables_factor))
            state_pretty_factor = list(zip(state_names_factor, state_bits_factor))
            state_pretty_discrete = [state_names_discrete[len(state_names_discrete) - 1 - i2]]
            state_pretty = state_pretty_factor + state_pretty_discrete
            state_values_pretty.append((state_pretty, state_value))

            state_decision = state_decisions[i1, i2]
            decision_pretty = decision_names[len(decision_names) - 1 - state_decision]
            state_decisions_pretty.append((state_pretty, [decision_pretty]))

    return state_values_pretty, state_decisions_pretty


def value_iteration_both(nb_states_factor, nb_states_discrete, nb_decisions, transition_probabilities, utility_values,
                         state_variable_ordering_factor, state_variable_ordering_discrete):
    time1 = time.perf_counter()
    gamma = 0.9  # TODO make gamma customizable
    epsilon = 0.00001  # TODO make epsilon customizable
    state_values = dict(((i1, i2), 0) for i1 in range(0, nb_states_factor) for i2 in range(0, nb_states_discrete))
    state_decisions = dict(((i1, i2), None) for i1 in range(0, nb_states_factor) for i2 in range(0, nb_states_discrete))
    converged = False
    nb_iterations = 0
    while not converged:
        nb_iterations += 1
        converged = True
        for i1 in range(0, nb_states_factor):
            for i2 in range(0, nb_states_discrete):
                max_value = None
                decision = None
                for j in range(0, nb_decisions):
                    utility = utility_values[i1, i2, j]
                    next_state_value = expected_next_state_value_both(i1, i2, j, transition_probabilities, state_values,
                                                                      state_variable_ordering_factor,
                                                                      state_variable_ordering_discrete)
                    value = utility + gamma * next_state_value
                    if max_value is None:
                        max_value = value
                        decision = j
                    elif value > max_value:
                        max_value = value
                        decision = j

                if abs(state_values[i1, i2] - max_value) > epsilon:
                    converged = False

                state_values[i1, i2] = max_value
                state_decisions[i1, i2] = decision

    time2 = time.perf_counter()
    vi_time = time2 - time1
    print('Value iteration time: ', vi_time, ' in ', nb_iterations, ' iterations')
    return state_values, state_decisions


# TODO recursive probability calculations need to be done only once after which they can be stored
def expected_next_state_value_both(i1, i2, j, transition_probabilities, state_values,
                                   state_variable_ordering_factor, state_variable_ordering_discrete):
    return expected_next_state_value_rec_both(i1, i2, j, transition_probabilities, state_values,
                                              state_variable_ordering_factor, state_variable_ordering_discrete, 0, 0)


def expected_next_state_value_rec_both(i1, i2, j, transition_probabilities, state_values,
                                       state_variable_ordering_factor, state_variable_ordering_discrete,
                                       state_factor, variable):
    if len(state_variable_ordering_factor) > variable:
        transition_probability = transition_probabilities[i1, i2, j][state_variable_ordering_factor[variable]]
        if transition_probability == 0:
            return (1 - transition_probability) \
                   * expected_next_state_value_rec_both(i1, i2, j, transition_probabilities, state_values,
                                                        state_variable_ordering_factor, state_variable_ordering_discrete,
                                                        state_factor * 2, variable + 1)

        elif transition_probability == 1:
            return transition_probability \
                   * expected_next_state_value_rec_both(i1, i2, j, transition_probabilities, state_values,
                                                        state_variable_ordering_factor, state_variable_ordering_discrete,
                                                        state_factor * 2 + 1, variable + 1)

        else:
            return transition_probability \
                   * expected_next_state_value_rec_both(i1, i2, j, transition_probabilities, state_values,
                                                        state_variable_ordering_factor, state_variable_ordering_discrete,
                                                        state_factor * 2 + 1, variable + 1) + \
                   (1 - transition_probability) \
                   * expected_next_state_value_rec_both(i1, i2, j, transition_probabilities, state_values,
                                                        state_variable_ordering_factor, state_variable_ordering_discrete,
                                                        state_factor * 2, variable + 1)

    else:
        return expected_next_state_value_both_discrete_part(i1, i2, j, transition_probabilities, state_values,
                                                            state_variable_ordering_discrete, state_factor)


def expected_next_state_value_both_discrete_part(i1, i2, j, transition_probabilities, state_values,
                                                 state_variable_ordering_discrete, state_factor):
    value = 0
    total_prob = 0
    for s in range(0, len(state_variable_ordering_discrete)):
        transition_probability = transition_probabilities[i1, i2, j][state_variable_ordering_discrete[s]]
        total_prob += transition_probability
        state_value = state_values[state_factor, len(state_variable_ordering_discrete) - 1 - s]  # TODO correct entry?
        value += transition_probability*state_value

    # if total_prob != 0:
    #     value = value * (1 / total_prob)  # TODO should we account for incorrect accumulated probability?

    return value


def get_transition_probabilities_both(formula, evidence, utilities, next_state_variables,
                                      transition_probabilities, utility_values, i1, i2, j):
    result = formula.evaluate(weights=evidence)
    score = 0.0
    for r in result:
        vpos = result[r]
        vneg = 1.0 - result[r]
        if r in utilities:
            score += vpos * float(utilities.get(r, 0.0))
            score += vneg * float(utilities.get(-r, 0.0))

        elif r in next_state_variables:
            transition_probabilities[(i1, i2, j)][r] = vpos

    utility_values[(i1, i2, j)] = score
    return score


def value_iteration_factor(nb_states, nb_decisions, transition_probabilities, utility_values, state_variable_ordering):
    time1 = time.perf_counter()
    gamma = 0.9  # TODO make gamma customizable
    epsilon = 0.00001  # TODO make epsilon customizable
    state_values = dict((i, 0) for i in range(0, nb_states))
    state_decisions = dict((i, None) for i in range(0, nb_states))
    converged = False
    nb_iterations = 0
    while not converged:
        nb_iterations += 1
        converged = True
        for i in range(0, nb_states):
            max_value = None
            decision = None
            for j in range(0, nb_decisions):
                utility = utility_values[i, j]
                next_state_value = expected_next_state_value2(i, j, transition_probabilities, state_values,
                                                              state_variable_ordering)  # TODO version 2 more efficient?
                value = utility + gamma * next_state_value
                if max_value is None:
                    max_value = value
                    decision = j
                elif value > max_value:
                    max_value = value
                    decision = j

            if abs(state_values[i] - max_value) > epsilon:
                converged = False

            state_values[i] = max_value
            state_decisions[i] = decision

    time2 = time.perf_counter()
    vi_time = time2 - time1
    print('Value iteration time: ', vi_time, ' in ', nb_iterations, ' iterations')
    return state_values, state_decisions


def expected_next_state_value(i, j, transition_probabilities, state_values, state_variable_ordering):
    return expected_next_state_value_rec(i, j, transition_probabilities, state_values, state_variable_ordering, 0)


def expected_next_state_value_rec(i, j, transition_probabilities, state_values, state_variable_ordering, nb):
    if state_variable_ordering:
        transition_probability = transition_probabilities[i, j][state_variable_ordering[0]]
        new_ordering = state_variable_ordering[1:]
        return transition_probability*expected_next_state_value_rec(i, j, transition_probabilities,
                                                                    state_values, new_ordering, nb*2 + 1) + \
            (1-transition_probability)*expected_next_state_value_rec(i, j, transition_probabilities,
                                                                     state_values, new_ordering, nb*2)
    else:
        return state_values[nb]


def expected_next_state_value2(i, j, transition_probabilities, state_values, state_variable_ordering):
    return expected_next_state_value_rec2(i, j, transition_probabilities, state_values, state_variable_ordering, 0, 0)


def expected_next_state_value_rec2(i, j, transition_probabilities, state_values, state_variable_ordering, state, variable):
    if len(state_variable_ordering) > variable:
        transition_probability = transition_probabilities[i, j][state_variable_ordering[variable]]
        if transition_probability == 0:
            return (1 - transition_probability) * expected_next_state_value_rec2(i, j, transition_probabilities,
                                                                                 state_values, state_variable_ordering,
                                                                                 state * 2, variable + 1)

        elif transition_probability == 1:
            return transition_probability * expected_next_state_value_rec2(i, j, transition_probabilities,
                                                                           state_values, state_variable_ordering,
                                                                           state * 2 + 1, variable + 1)

        else:
            return transition_probability * expected_next_state_value_rec2(i, j, transition_probabilities,
                                                                           state_values, state_variable_ordering,
                                                                           state * 2 + 1, variable + 1) + \
                   (1 - transition_probability) * expected_next_state_value_rec2(i, j, transition_probabilities,
                                                                                 state_values, state_variable_ordering,
                                                                                 state * 2, variable + 1)

    else:
        return state_values[state]


def solve_mdp_discrete(formula, decisions, current_state_variables, utilities, next_state_variables, constraints):
    counter = 0  # TODO test counter
    prob_time2 = 0  # TODO test time
    time1 = time.perf_counter()
    state_ids, state_names = zip(*current_state_variables)
    state_variable_ordering = get_variable_ordering2(state_names)
    decision_ids, decision_names = zip(*decisions)

    # TODO range is now linear instead of exponential
    transition_probabilities = dict(((i, j), dict((k, 0) for k in next_state_variables))
                                    for i in range(0, len(current_state_variables))
                                    for j in range(0, len(decisions)))
    utility_values = dict(((i, j), 0)
                          for i in range(0, len(current_state_variables))
                          for j in range(0, len(decisions)))

    # TODO range is now linear instead of exponential
    for i in range(0, len(current_state_variables)):
        # TODO now need to only iterate over binary numbers where only one bit is 1
        state = num2bits(1 << i, len(current_state_variables))
        evidence_state = dict(zip(state_names, map(int, state)))

        constraints_ok = True
        for c in constraints:
            if not c.check(dict(zip(state_ids, map(int, state)))):
                constraints_ok = False
                break
        if not constraints_ok:
            continue

        # TODO range is now linear instead of exponential
        for j in range(0, len(decisions)):
            # TODO now need to only iterate over binary numbers where only one bit is 1
            choices = num2bits(1 << j, len(decisions))
            evidence_decisions = dict(zip(decision_names, map(int, choices)))
            evidence = evidence_decisions.copy()
            evidence.update(evidence_state)

            constraints_ok = True
            for c in constraints:
                if not c.check(dict(zip(decision_ids, map(int, choices)))):
                    constraints_ok = False
                    break
            if not constraints_ok:
                continue

            # TODO correct use of constraints?
            constraints_ok = True
            for c in constraints:
                if not c.check(dict(zip(state_ids + decision_ids, map(int, state + choices)))):
                    constraints_ok = False
                    break
            if not constraints_ok:
                continue

            counter += 1  # TODO test counter increment
            timet1 = time.perf_counter()  # TODO test time
            get_transition_probabilities(formula, evidence_state, evidence_decisions, utilities, next_state_variables,
                                         transition_probabilities, utility_values, i, j)
            timet2 = time.perf_counter() - timet1  # TODO test time
            prob_time2 += timet2  # TODO test time

    time2 = time.perf_counter()
    prob_time = time2 - time1
    print('Probability calculation time: ', prob_time)
    print('Actual Probability calculation time: ', prob_time2)
    print('Number of evaluations: ', counter)
    # TODO nb of states and decisions is now linear instead of exponential
    state_values, state_decisions = value_iteration_discrete(len(current_state_variables), len(decisions),
                                                             transition_probabilities, utility_values,
                                                             state_variable_ordering)
    state_values_pretty = []
    state_decisions_pretty = []
    for i in range(0, len(current_state_variables)):
        state_value = state_values[i]
        # state_bits = num2bits(1 << i, len(current_state_variables))  # TODO correct conversion?
        # state_pretty = zip(state_names, state_bits)
        state_pretty = state_names[len(state_names) - 1 - i]
        state_values_pretty.append(([state_pretty], state_value))

        state_decision = state_decisions[i]
        # decision_bits = num2bits(1 << state_decision, len(decisions))  # TODO correct conversion?
        # decision_pretty = zip(decision_names, decision_bits)
        decision_pretty = decision_names[len(decision_names) - 1 - state_decision]
        # state_pretty2 = zip(state_names, state_bits)
        state_pretty2 = state_names[len(state_names) - 1 - i]
        state_decisions_pretty.append(([state_pretty2], [decision_pretty]))

    return state_values_pretty, state_decisions_pretty


def value_iteration_discrete(nb_states, nb_decisions, transition_probabilities, utility_values, state_variable_ordering):
    time1 = time.perf_counter()
    gamma = 0.9  # TODO make gamma customizable
    epsilon = 0.00001  # TODO make epsilon customizable
    state_values = dict((i, 0) for i in range(0, nb_states))
    state_decisions = dict((i, None) for i in range(0, nb_states))
    converged = False
    nb_iterations = 0
    while not converged:
        nb_iterations += 1
        converged = True
        for i in range(0, nb_states):
            max_value = None
            decision = None
            for j in range(0, nb_decisions):
                utility = utility_values[i, j]
                # TODO expected state value is calculated differently
                next_state_value = expected_next_state_value_discrete(i, j, transition_probabilities, state_values,
                                                                      state_variable_ordering)
                value = utility + gamma * next_state_value
                if max_value is None:
                    max_value = value
                    decision = j
                elif value > max_value:
                    max_value = value
                    decision = j

            if abs(state_values[i] - max_value) > epsilon:
                converged = False

            state_values[i] = max_value
            state_decisions[i] = decision

    time2 = time.perf_counter()
    vi_time = time2 - time1
    print('Value iteration time: ', vi_time, ' in ', nb_iterations, ' iterations')
    return state_values, state_decisions


def expected_next_state_value_discrete(i, j, transition_probabilities, state_values, state_variable_ordering):
    value = 0
    total_prob = 0
    for s in range(0, len(state_variable_ordering)):
        transition_probability = transition_probabilities[i, j][state_variable_ordering[s]]
        total_prob += transition_probability
        state_value = state_values[len(state_variable_ordering) - 1 - s]  # TODO correct entry?
        value += transition_probability*state_value

    # if total_prob != 0:
    #     value = value * (1 / total_prob)  # TODO should we account for incorrect accumulated probability?

    return value


def get_transition_probabilities(formula, evidence_state, evidence_decisions, utilities, next_state_variables,
                                 transition_probabilities, utility_values, i, j):
    evidence = evidence_decisions.copy()
    evidence.update(evidence_state)
    result = formula.evaluate(weights=evidence)
    score = 0.0
    for r in result:
        vpos = result[r]
        vneg = 1.0 - result[r]
        if r in utilities:
            score += vpos * float(utilities.get(r, 0.0))
            score += vneg * float(utilities.get(-r, 0.0))

        elif r in next_state_variables:
            transition_probabilities[(i, j)][r] = vpos

    utility_values[(i, j)] = score
    return score


def mdpproblog(model, koption=None, **kwargs):
    """Evaluate an mdp ProbLog model

    :param model: ProbLog model
    :type model: problog.logic.LogicProgram
    :param koption: specifies knowledge compilation tool (omit for system default)
    :return: optimal value for each state, optimal action for each state
    """

    time1 = time.perf_counter()

    eng = DefaultEngine(label_all=True)
    db = eng.prepare(model)

    decisions_t = set(d[0] for d in eng.query(db, Term('action', None)))
    utilities = dict(eng.query(db, Term("utility", None, None)))

    state_factors = [s[0] for s in eng.query(db, Term("state_variable", None))]
    current_state_factors = create_state_variables2(state_factors, 'current')
    next_state_factors = create_state_variables2(state_factors, 'next')

    discrete_states = [s[0] for s in eng.query(db, Term("nb_state_variable", None))]
    current_discrete_states = create_state_variables2(discrete_states, 'current')
    next_discrete_states = create_state_variables2(discrete_states, 'next')

    # TODO actions and state variable clauses added correctly?
    for d in decisions_t:
        db += d.with_probability(Constant("?"))

    for f in current_state_factors:
        db += f.with_probability(Constant("?"))

    for s in current_discrete_states:
        db += s.with_probability(Constant("?"))

    gp = eng.ground_all(db, target=None,
                        queries=set(utilities.keys()) | set(next_state_factors) | set(next_discrete_states))

    actions = []
    decision_nodes = set()
    states_discrete = []
    states_factor = []
    for i, n, t in gp:
        if t == "atom" and n.probability == Term("?"):
            if n.name in decisions_t:
                actions.append((i, n.name))
                decision_nodes.add(i)

            if n.name in current_state_factors:
                states_factor.append((i, n.name))

            if n.name in current_discrete_states:
                states_discrete.append((i, n.name))

    constraints = []
    for c in gp.constraints():
        if set(c.get_nodes()) & decision_nodes:
            constraints.append(c)

    if decision_nodes:
        knowledge = get_evaluatable(koption).create_from(gp)
        time2 = time.perf_counter()
        comp_time = time2 - time1
        print('Knowledge compilation time: ', comp_time)
        if not states_factor:
            result = solve_mdp_discrete(knowledge, actions, states_discrete, utilities, next_discrete_states, constraints)
        elif not states_discrete:
            result = solve_mdp_factor_single_decision(knowledge, actions, states_factor, utilities, next_state_factors, constraints)
        else:
            result = solve_mdp_both_single_decision(knowledge, actions, states_factor, states_discrete, utilities,
                                                    next_state_factors, next_discrete_states, constraints)
    else:
        result = [], []

    return result


def num2bits(n, nbits):
    bits = [False] * nbits
    for i in range(1, nbits + 1):
        bits[nbits - i] = bool(n % 2)
        n >>= 1
    return bits


if __name__ == "__main__":
    main(sys.argv)
