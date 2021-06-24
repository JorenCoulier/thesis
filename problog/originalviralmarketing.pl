% state
state_variable(marketed(P)) :- person(P).

% actions
%action(market(P)) :- person(P).
%action(market(none)).
action(market(P)) :- person(P).
action(market(none)).

% utility
utility(market(P), -1) :- person(P).
utility(buys(P), 5) :- person(P).

% transition model
next(marketed(X)) :- market(X).
0.5::next(marketed(X)) :- not(market(X)), current(marketed(X)).

% reward model
0.2::buy_from_marketing(P) :- person(P).
0.3::buy_from_trust(P) :- person(P).
buys(X) :- current(marketed(X)), buy_from_marketing(X).
buys(X) :- trusts(X,Y), buys(Y), buy_from_trust(X).

% Background knowledge
person(fabio).
person(leliane).
trusts(fabio,leliane).
trusts(leliane,fabio).

person(denis).
trusts(denis,leliane).
trusts(denis,fabio).

person(thiago).
trusts(leliane,thiago).
trusts(thiago,leliane).

%person(one).
%trusts(one, leliane).
%trusts(thiago, one).

%person(two).
%trusts(two, one).
%trusts(fabio, two).

%person(three).
%trusts(three, denis).
%trusts(one, three).