:- use_module(library(lists)).

nb(X) :- X is 5.

%state features
%h means owner has cup
%c means robot has cup
state_variable(X) :- member(X, [h, c]).

%discrete states
nb_state_variable(loc(X)) :- nb(Nb), between(0,Nb,X).

% actions
action(X) :- member(X, [goCafe, goOffice, buyC, delC, nothing]).

% transition model
next(loc(X)) :- current(loc(X)), \+ goCafe, \+ goOffice.
%0.8::next(location(Nb2)); 0.2::next(location(Nb1)) :- current(location(Nb1)), Nb1 > 0, do(goCafe), Nb2 is Nb1 - 1.
%0.8::next(location(Nb2)); 0.2::next(location(Nb1)) :- nb(Nb), current(location(Nb1)), Nb1 < Nb, do(goOffice), Nb2 is Nb1 + 1.
0.8::next(loc(Nb2)) :- current(loc(Nb1)), Nb1 > 0, goCafe, Nb2 is Nb1 - 1.
0.8::next(loc(Nb2)) :- nb(Nb), current(loc(Nb1)), Nb1 < Nb, goOffice, Nb2 is Nb1 + 1.
0.2::next(loc(Nb1)) :- current(loc(Nb1)), Nb1 > 0, goCafe.
0.2::next(loc(Nb1)) :- nb(Nb), current(loc(Nb1)), Nb1 < Nb, goOffice.
%next(location(Nb2)) :- current(location(Nb1)), Nb1 > 0, do(goCafe), Nb2 is Nb1 - 1.
%next(location(Nb2)) :- nb(Nb), current(location(Nb1)), Nb1 < Nb, do(goOffice), Nb2 is Nb1 + 1.

%\+ do(goCafe) :- current(location(0)).
%\+ do(goOffice) :- nb(Nb), current(location(Nb)).

next(h) :- \+ current(h), nb(Nb), current(loc(Nb)), current(c), delC.
\+ next(h) :- current(h).

next(c) :- current(loc(0)), buyC.
next(c) :- current(c), \+ delC.
\+ next(c) :- delC.

% utility
utility(current(h), 0.9).
utility(nothing, 0.001).