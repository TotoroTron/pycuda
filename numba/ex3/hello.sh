#!/bin/bash

# "#!" : shebang : tells the OS to execute the script using specified interpreter
# "#!/bin/bash" : tells OS to execute the script using Bash shell
# "#!/usr/bin/python3" : tells OS to execute the script using python


readonly const=2024
echo "Hello World $const!"


# CHECK IF NO PARAMS
if [[ -z $1 ]]; # if param is NULL or uninit (no arg passed)
then
	echo "No parameter passed. Exiting."
	exit 1
fi

# CHECK IF NUMBER OF ARGS EQUALS 2
if [[ $# -ne 2 ]]; # if 'num params' not equal 2
then
	echo "Exactly two integers are required. Exiting."
	exit 1
fi


# CHECK IF PARAMS ARE INTEGERS
pattern='^-?[0-9]+$' # regex paittern for int
# ^ anchor match at start of string
# -? match optional minus sign. ? means "one or zero" of preceding char
# [0-9]+ match one or more digits. + means "one or more" of preceding element
# $ anchor match at end of string 
if [[ $1 =~ $pattern  &&  $2 =~ $pattern ]];
then
	echo "Both args are integers."
	echo "Parameter 1 = $1"
	echo "Parameter 2 = $2"
else
	echo "Both args must be integers. Exiting."
	exit 1
fi



a=$1 # assign var a = arg 2
b=$2 # assign var b = arg 1
p=$(($a*$b))
echo "The product of $a and $b = $p"

# Create output dir if doesnt exist already
output_dir="hello_out"
mkdir -p $output_dir

# Create filename and file
filename="${1}_${2}_out.txt"
filepath="${output_dir}/${filename}"

# Append some text to 1st line of file
echo "The product of $a and $b = $p" > $filepath

# Append some text to 2nd line of file
echo "Hello again!" >> $filepath

# Single > means overwrite with new content if file exists already
# Double >> means append new content to end of an existing file

# Append a bunch of lines with loop
for i in {2..20};
do
	echo "Hello again x$i combo!" >> $filepath
done


echo "Created file: $filepath containing result."
