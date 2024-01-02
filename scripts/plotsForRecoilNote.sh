#!/bin/bash

baseDir=/eos/cms/store/cmst3/user/psilva/Wmass/RecoilTest_regress-models

#prepare output
tag=`date "+%d%b"`
tag=24Aug
outDir=~/www/Wmass/${tag}
mkdir -p ${outDir}
cp ~/www/index.php ${outDir}

#
# evolution of the loss
#

loss() {
    for i in 0 100 300; do
        inputs="model${i}":${baseDir}/model${i}_lne1_e2
        python scripts/makeTrainValidationPlots.py ${inputs}
        for ext in png pdf; do
            mv loss.${ext} ${outDir}/model${i}_loss.${ext};
        done
    done
    rm *.{png,pdf}


    inputs="0":${baseDir}/model300_lne1_e2,"0.1":${baseDir}/model301_lne1_e2,"0.2":${baseDir}/model302_lne1_e2
    python scripts/makeTrainValidationPlots.py ${inputs}
    for ext in png pdf; do
        mv loss.${ext} ${outDir}/dropoutvars_loss.${ext};
    done
    rm *.{png,pdf}


    inputs="32x16x4":${baseDir}/model301_lne1_e2,"64x32x16":${baseDir}/model303_lne1_e2,"64x32x16x4":${baseDir}/model315_lne1_e2
    python scripts/makeTrainValidationPlots.py ${inputs}
    for ext in png pdf; do
        mv loss.${ext} ${outDir}/depthvars_loss.${ext};
    done
    rm *.{png,pdf}

    inputs="[32x16x4]":${baseDir}/model301_lne1_e2,"[64][32x16x4]":${baseDir}/model307_lne1_e2,"[128x16][32x16x4]":${baseDir}/model313_lne1_e2
    echo $inputs
    python scripts/makeTrainValidationPlots.py ${inputs}
    for ext in png pdf; do
        mv loss.${ext} ${outDir}/preprocvars_loss.${ext};
    done
    rm *.{png,pdf}

}

#
#fit parameters
#
fit_params() {
    for p in lne1 e2; do
        python scripts/drawParametersFit.py model300_lne1_e2 ${p};
    done
    mv *.{png,pdf} ${outDir}
}

#
# ideograms and performance benchmarks
#
ideograms() {
    for m in 310; do #0 100 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320; do         
        python scripts/makeIdeogramPlots.py -i ${baseDir}/model${m}_lne1_e2/train -o ${outDir}/ideograms${m}_lne1_e2.root -n 2000 &
    done
    #for vlist in vlist1 vlist2; do
    #    python scripts/makeIdeogramPlots.py -i ${baseDir}/model310_lne1_e2_${vlist}/train -o ${outDir}/ideograms${m}_lne1_e2_${vlist}.root -n 200000 &
    #done
}

draw_ideograms() {
    python scripts/analyzeIdeogramPlots.py -i ~/www/Wmass/24Aug/ -o ~/www/Wmass/24Aug/ -l base
    python scripts/analyzeIdeogramPlots.py -i ~/www/Wmass/24Aug/ -o ~/www/Wmass/24Aug/ -l sparam
    python scripts/analyzeIdeogramPlots.py -i ~/www/Wmass/24Aug/ -o ~/www/Wmass/24Aug/ -l vlist
}

#loss
fit_params
#ideograms
#draw_ideograms