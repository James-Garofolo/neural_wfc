# Neural Generation of Tile-Interactivity Rule Sets for Wave Function Collapse

The goal of this project is to train an artificial neural network to produce a rule set for the Wave Function Collapse algorithm (https://github.com/mxgmn/WaveFunctionCollapse.git). 

okay so imma make this readme for you and i for right now jordan, since i can't really think of what to write for anyone else that sees it right now. 

the way i see this project, there's 8 parts to this project

* a script to split the game map into individual rooms (this doesn't need to be very usable because i can't imagine many other games needing this) jordan 
* a classifier that recognizes each tile jordan
  * this could be a neural network if we wanted to go easy mode, but i think it would be cool to try and use k means clustering with something like silhouette             detection so we don't have to sit there and manually grab a screenshot of every single tile
  * we wanna use this to turn each of the rooms into a matrix of 1-hot vectors for training data
* a data prepping function that adds unknown blocks to the maps jim
  * basically make a good 10-20 copies of each room matrix, and randomly replace a varying amount of the tiles in each room with the "undecided" token
  * i think the percentage drop rate should be linearly varied from 0% to like 90% as we generate more examples so the distribution of placed tile count in the             training set is uniform, but we also have to make sure that none of the training examples have literally no tiles, or that wouldn't be helpful. 
  * very important that the training data gets order-shuffled, so it gets to see an example of each room at least once, otherwise it won't know how to use a kind of       tile
* the actual wave function collapse algorithm jordan
  * for unit testing's sake, i think it'd be good to make this solve sudoku puzzles first
  * good way to test the entropy collapse condition is to have it just start with a naked puzzle and fill in numbers at random until a unique solution arises
  * see if it gets stuck. i'm not actually sure that it can, but i'm also not fully convinced that it can't. if it can't, we definitely need a condition that undoes         stuff until a solution becomes possible again
* the rule-generating network jim
  * absolute bare-bones necessity is just a feed-forward guy with an input space the same size as the room
  * better option is probably some kind of transformer architecture that can handle whatever sized input we want
* a normal training/evaluation script that uses the training data like you would with any other network jim
* a script that has it mass-generate rooms from just the walls or from nothing and saves them as png/csv files, because that's how you'd want to use it in practice jordan
  * i think we should use this to evaluate how well it learned the patterns qalitatively
  * obv this would also be where we get figures for the paper from
  * we could also design a few tests to quantify the usability of the outputs, i.e. does it do dumb things like put walls right in front of doors, make areas               impossible to get to for no reason, etc
* for style points, a visualization script that lets you watch it build a room from empty, or maybe even a whole dungeon jim
  * could totally do this using pygame, which i'm pretty good at, unless you have other ideas
  * if i get the dynamic input size thing working, it'd be pretty cool to try and make a version that generates whole dungeons by default
  * we could totally generate a dungeon using the static input size guy too, we'd just need to mandate that the wall state of each room is carried over to the rooms       next to it and only generate rooms that doors lead to

i figure we can just one-by-one claim four each, and that'll be a fair way of divvying stuff up
