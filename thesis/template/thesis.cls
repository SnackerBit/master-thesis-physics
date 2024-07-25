 \NeedsTeXFormat{LaTeX2e}[1995/12/01] % TODO: What does this command do?
 \ProvidesClass{template/thesis}[2024/04/18 thesis layout]
 
 % Basisklasse: KOMA-Buch (scrbook)
\PassOptionsToPackage{headinclude=true,footinclude=false}{typearea}
\LoadClass[version=last,		% use latest version
	headinclude=true			% has something to do with spacing and borders
	,footinclude=false			%
	,fontsize=11pt				% font size
	,twocolumn=false			% single column for thesis, two columns for papers
	,twoside=false				% TODO: Why is this different to below??
	,open=any					% TODO: Why is this different to below??
	,footnotes=multiple			% behaviour for multiple footnotes at the same place
	,chapterprefix=true			% alignment of chapter titles
	,appendixprefix=true		% alignment of appendix titles
	,numbers=noenddot			% dot for table of contents
	,bibliography=totocnumbered	% style of bibliography
	]{scrbook}
\KOMAoptions{BCOR=1cm	% binding correction (space at the middle of page for binding)
		,DIV=10			% factor for horizontal and vertical mirroring of pages
		,open=right     % Where new chapters begin (on the right/left side of a page)
		,fontsize=11pt  % font size
		,twoside=true   % thesis should be two-sided
		}	

% Easy colors
\RequirePackage{xcolor}

% better sections + centering
%\RequirePackage{sectsty}
		
% PDF comments for TODOS
\RequirePackage{pdfcomment}
\newcommand{\todo}[1]{
	\pdfcomment[color=red,open=true]{#1}
}

% Use biblatex for the bibliography
\PassOptionsToPackage{
	sorting=none,
	autolang=other,
	backend=biber,
	bibstyle=numeric-comp,
	citestyle=numeric-comp,
}{biblatex}
\RequirePackage{biblatex}
\IfFileExists{\jobname.bib}{\addbibresource{\jobname.bib}}{\relax}
\defbibheading{bibliography}[\bibname]{\chapter{#1}}


% AMS packages for equations and math symbols
\PassOptionsToPackage{intlimits}{amsmath}		% integral limits below and above integrals
\RequirePackage{amsmath}						% Math symbols
\RequirePackage{amstext}						% text in math mode with \text{}
\RequirePackage{nicefrac}						% For having nice fractions of the form a/b, e.g. for units: m/s etc.
% bold math package: Use bold symbols in math mode
\RequirePackage{bm}
\PassOptionsToPackage{showonlyrefs}{mathtools}	% Only show equation number if the equation is referenced somewhere in the document
\RequirePackage{mathtools}						% Advanced functionality for amsmath.
% Package for subfigures
\RequirePackage{subcaption}
\PassOptionsToPackage{
	font=small,
	labelfont=bf	
}{subcaption}


% Command for linking figures (different link color)
\newcommand*{\figref}[1]{%
	\begingroup
	\hypersetup{
		linkcolor=blue,
		linkbordercolor=blue,
	}%
	\ref{#1}%
	\endgroup
}

% More useful packages
\RequirePackage{booktabs}	% nice tables
\RequirePackage{caption}	% nice captions for figures, tables etc.
\RequirePackage{multirow}	% enables support for having multirows in tables
\RequirePackage{xspace}		% optional space (Tries to only put space if needed). Useful for custom commands	
\RequirePackage{babel}		% Support for language specific rules
\RequirePackage{csquotes}	% Necessary for correct typesetting of quotes

% Microtype makes text generally more pleasent
\PassOptionsToPackage{babel}{microtype}
\RequirePackage{microtype}
\DisableLigatures{encoding=T1,family=tt*}

% Package for table of content
\RequirePackage{tocbasic}

% Some abbreviations as commands
\newcommand{\eg}{e.g.\xspace}
\newcommand{\wrt}{w.r.t.\xspace}

% Some special symbols as commands
\RequirePackage{amssymb}
\DeclareMathAlphabet{\mathbbb}{U}{bbold}{m}{n}
\DeclareMathOperator{\Tr}{Tr}
\let\Re\relax
\DeclareMathOperator{\Re}{Re}
\let\Im\relax
\DeclareMathOperator{\Im}{Im}
\DeclareMathOperator{\argmax}{argmax}
\DeclareMathOperator{\argmin}{argmin}
\DeclareMathOperator{\qf}{qf}
%\DeclareMathOperator{\mod}{mod}
\newcommand{\id}{\mathbbb{1}}
\newcommand{\iu}{\mathrm{i}}% imaginary unit number i
\newcommand{\Stiefel}{\text{St}(n,p)}

% Setup for hyperlinks and pdf metadata
\RequirePackage{hyperref}
\AtBeginDocument{
	\hypersetup{
		urlcolor=blue,
		pdfauthor={\@author},
		pdftitle={\@title},
		pdfsubject={\@subject},
		pdfkeywords={Technische Universit\"at M\"unchen;Fakult\"at f\"ur Physik;\@subject
		}
	}
}

% Code to display header lines with the current chapter at the top of each page
\RequirePackage[ilines,headsepline]{scrlayer-scrpage}
\pagestyle{scrheadings}
\newcommand\OrigBackmatter{} \let\OrigBackmatter=\backmatter
\renewcommand{\backmatter}{\OrigBackmatter \pagestyle{scrplain}}

% Layout of the title page
\newcommand*{\othertitle@font}{\usekomafont{subtitle}\mdseries}
\renewcommand*{\maketitle}{
	\begin{titlepage}
		\noindent\begin{minipage}[b]{.5\linewidth}
			\normalfont\sffamily Technische Universit\"{a}t M\"{u}nchen\\
			Fakult\"{a}t f\"{u}r Physik
		\end{minipage}
		\hfill
		\begin{minipage}[b]{.4\linewidth}
			\raggedleft{\includegraphics[width=1cm]{template/pictures/PH}\hspace{.5cm}\includegraphics[width=1.87cm]{template/pictures/TUM}}{}
		\end{minipage}
		\null\vfill
		\begin{center}
			\ifx\@subject\@empty \else
				{\subject@font \@subject \par}
				\vskip 3em
			\fi
			{\titlefont\huge \@title\par}
			\vskip 1em
			{\ifx\@subtitle\@empty\else\usekomafont{subtitle}\@subtitle\par\fi}%
			\vskip 2em
			{\othertitle@font \lineskip 0.75em
				\begin{tabular}[t]{c}
					\@author
				\end{tabular}\par
			}
			\vskip 1.5em
			{\othertitle@font \@date \par}%
			\vskip \z@ \@plus3fill
			{\othertitle@font \@publishers \par}%
			\vskip 3em
		\end{center}\par
		\@thanks
		\vfill\null
		\if@twoside\next@tpage
		\begin{minipage}[t]{\textwidth}
			\@uppertitleback
		\end{minipage}\par
		\vfill
		\begin{minipage}[b]{\textwidth}
			\@lowertitleback
		\end{minipage}
		\fi
		\ifx\@dedication\@empty \else
			\next@tpage\null\vfill
			{\centering \Large \@dedication \par}%
			\vskip \z@ \@plus3fill
			\if@twoside \next@tpage\cleardoubleemptypage \fi
		\fi
	\end{titlepage}
}

% Fix table of content depth
\setcounter{tocdepth}{1}
% Fix subsection numbering
\setcounter{secnumdepth}{3}
% Set indent for captions to zero
\setcapindent{0pt}

\renewcommand{\bottomfraction}{0.5}			% Percentage of page space that figures etc can take at the bottom of pages
\renewcommand{\floatpagefraction}{0.7}		% Percentage of float object page that must be full before a new page is created
\renewcommand{\dblfloatpagefraction}{0.7}	% The same but for two column texts

\tolerance 1414				% Fine tune paragraph breaking
\hbadness 1414				% Influences user report about line breaking
\emergencystretch 1.5em		% also used for line breaking
\hfuzz 0.3pt
\widowpenalty=10000
\vfuzz \hfuzz
\raggedbottom				% Creates more even spacing by sometimes increasing space between paragraphs

% PLOTS
\RequirePackage{pgfplots}
%\pgfplotsset{compat=1.18}
\usepgfplotslibrary{colorbrewer}
\RequirePackage{tikz}
% A single subfigure
\def\singleFigureWidth{8 cm}
\def\singleFigureHeight{4.944 cm}
\def\insetFigureWidth{4 cm}
\def\insetFigureHeight{2.2 cm}
% Two subfigures stacked ontop of each other
\def\doubleVerticalFigureWidth{11 cm}
\def\doubleVerticalFigureHeight{5 cm}

% 3 subfigures on top of each other
\def\tripleVerticalFigureWidth{11 cm}
\def\tripleVerticalFigureHeight{4 cm}

% 4 subfigures aranged in a 2x2 pattern
\def\twoByTwoFigureWidth{6.75cm}
\def\twoByTwoFigureHeight{4.1717cm}

% 4 subfigures side by side on one page
\def\gsEnergyVsDtauFigureWidth{3.2cm}
\def\gsEnergyVsDtauFigureHeight{4cm}

% 3 lengthy subfigures below each other
\def\globalQuenchLargeFieldFigureWidth{12cm}
\def\globalQuenchLargeFieldFigureHeight{3cm}

\def\legendscale{1.0}

\def\doubleFigureWidth{6.125cm}
\def\doubleFigureHeight{4.635cm}
% PLOT Colors
\definecolor{7blue1}{HTML}{eff3ff}
\definecolor{7blue2}{HTML}{c6dbef}
\definecolor{7blue3}{HTML}{9ecae1}
\definecolor{7blue4}{HTML}{6baed6}
\definecolor{7blue5}{HTML}{4292c6}
\definecolor{7blue6}{HTML}{2171b5}
\definecolor{7blue7}{HTML}{084594}

\definecolor{5blue1}{HTML}{eff3ff}
\definecolor{5blue2}{HTML}{bdd7e7}
\definecolor{5blue3}{HTML}{6baed6}
\definecolor{5blue4}{HTML}{3182bd}
\definecolor{5blue5}{HTML}{08519c}

\definecolor{3blue1}{HTML}{6baed6}
\definecolor{3blue2}{HTML}{3182bd}
\definecolor{3blue3}{HTML}{08519c}

\definecolor{5orange1}{HTML}{feedde}
\definecolor{5orang2}{HTML}{fdbe85}
\definecolor{5orange3}{HTML}{fd8d3c}
\definecolor{5orange4}{HTML}{e6550d}
\definecolor{5orange5}{HTML}{a63603}

\definecolor{5red1}{HTML}{fee5d9}
\definecolor{5red2}{HTML}{fcae91}
\definecolor{5red3}{HTML}{fb6a4a}
\definecolor{5red4}{HTML}{de2d26}
\definecolor{5red5}{HTML}{a50f15}

\definecolor{3red1}{HTML}{fb6a4a}
\definecolor{3red2}{HTML}{de2d26}
\definecolor{3red3}{HTML}{a50f15}

\definecolor{5green1}{HTML}{edf8e9}
\definecolor{5green2}{HTML}{bae4b3}
\definecolor{5green3}{HTML}{74c476}
\definecolor{5green4}{HTML}{31a354}
\definecolor{5green5}{HTML}{006d2c}

\definecolor{5gray1}{HTML}{f7f7f7}
\definecolor{5gray2}{HTML}{cccccc}
\definecolor{5gray3}{HTML}{969696}
\definecolor{5gray4}{HTML}{636363}
\definecolor{5gray5}{HTML}{252525}

\definecolor{5purple1}{HTML}{f2f0f7}
\definecolor{5purple2}{HTML}{cbc9e2}
\definecolor{5purple3}{HTML}{9e9ac8}
\definecolor{5purple4}{HTML}{756bb1}
\definecolor{5purple5}{HTML}{54278f}