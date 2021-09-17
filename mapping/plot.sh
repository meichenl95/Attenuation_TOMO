#!/bin/bash

fln="evtstn"

gmt begin map pdf,png
gmt set COLOR_FOREGROUND RED4
gmt set COLOR_BACKGROUND BLUE4
gmt basemap -Rg -JA-150/55/45/5i -Bg
gmt coast -Wthinnest -Df -A10000
gawk '{print $2,$1}' $fln | uniq | gmt plot -Sa0.5c,black -Gred
gawk '{print $4,$3}' $fln | uniq | gmt plot -St0.1c,black -Gblue
gmt end

