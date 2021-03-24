- [How to Compile Flopy on Mac](#how-to-compile-flopy-on-mac)
  * [Preparation](#preparation)
  * [Install anaconda](#install-anaconda)
  * [Compile swtv4 file](#compile-swtv4-file)
- [Contaminant model brisbane airport](#contaminant-model-brisbane-airport)

# How to Compile Flopy on Mac
(Note: The installation of Anaconda is step 6-8. Ignore these steps if you already have python environments)
## Preparation


1. Make sure you have the CLT for Xcode on your Mac.
If not, try to type this command in your Terminal: 

`xcode-select -install`

or follow the installation instructions for Xcode CLT: 
https://www.ics.uci.edu/~pattis/common/handouts/macmingweclipse/allexperimental/macxcodecommandlinetools.html


2. Go to: /Applications/Utilities/Terminal to open terminal.
3. Install Homebrew (a package manager, more info: https://brew.sh) Copy-paste this command into your Terminal:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

(note: you have to copy every character, Once it finishes, type `brew` into your Terminal, if this message appears, go to next step.)

<div align=center><img width="400" height="400" src="https://user-images.githubusercontent.com/54530856/112263858-ee94f300-8caa-11eb-882b-af3b579d4ca6.png"></div>

4. Type `brew install gcc` (This step is time consuming depands on your internet condition.)
5. Type `gcc --version` to check if the installation of GCC success. (You should see the following message)

<div align=center><img width="400" alt="Screen Shot 2021-03-24 at 2 14 55 pm" src="https://user-images.githubusercontent.com/54530856/112264077-50edf380-8cab-11eb-9c36-5820843b2fdd.png"></div>

## Install anaconda
6. Run `brew install --cask anaconda` to install Anaconda
7. To setup the environment path, paste this command in your Terminal:

```
export PATH="/usr/local/anaconda3/bin:$PATH"
```

8. Restart your Terminal and type `spyder`, the IDLE should appear.

## Compile swtv4 file
9. Download the pymake by this link https://github.com/modflowpy/pymake
10. For those who don't know how to download files through Github:

<div align=center><img width="400" alt="Screen Shot 2021-03-24 at 2 27 29 pm" src="https://user-images.githubusercontent.com/54530856/112265203-12593880-8cad-11eb-922e-d9b4a822f761.png"></div>

11. Copy the pathname of the pymake-master and type `cd xxx` (xxx is your pathname just copied)
12. Run `python setup.py install`
13. Copy the pathname of the 'make_swtv4.py' (its in the 'examples' folder)
14. Run `python xxx` (xxx is your path name of make_swtv4.py). ***It will download the seawat source code and compile it on your computer***
15. Once it finishes, you will see the Unix executable file named 'swtv4' in pymake folder. 
16. Repeat step 15-16 for other scripts you need in the 'example' directory if needed.



# Contaminant model brisbane airport
During the period from 1988 to 2010 (22 years) it is estimated that approximately 700-750 L of a chemical leaked into ground at the site. The chemical release was not constant and is thought to have occurred over a 1- hour period at least once a month during the 22-year period in which it was used at the site. The chemical was released at the ground surface is thought to have rapidly infiltrated the soil profile and entered the underlying unconfined groundwater system. It is estimated that approximately 2.75 L per month was released at the site in this way (i.e., pulse loading of 2.75 L of chemical each month for the 22-year period from 1988 to 2010).

<div align=center><img width="400" height="400" src="https://user-images.githubusercontent.com/54530856/111432611-33171080-8738-11eb-971d-5bd576487999.png"></div>

<div align=center><img width="400" height="400" src="https://user-images.githubusercontent.com/54530856/111432632-3c07e200-8738-11eb-92f9-bf536816c5fd.png"></div>

<div align=center><img width="800" height="400" src="https://user-images.githubusercontent.com/54530856/111432642-3f9b6900-8738-11eb-9587-71c484e7c24e.png"></div>

