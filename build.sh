#!/bin/sh

nix-shell --run 'stack exec mysite clean && stack exec mysite build'
