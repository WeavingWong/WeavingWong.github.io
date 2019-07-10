#!/bin/sh
bundle exec jekyll build
rm _site/config.codekit3 _site/build.sh
rsync -auv _site WeavingWong@https://WeavingWong.github.io