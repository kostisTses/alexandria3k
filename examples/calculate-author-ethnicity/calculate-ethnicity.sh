#!/bin/bash

eth=$1

if [[ $# -ne 1 ]]; then 
    echo 'Usage: ./calculate-ethnicity.sh <eth>' 1>&2
    exit 1 
fi 

make ETH="$eth"