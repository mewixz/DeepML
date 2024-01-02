#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

TRAINPATH=/afs/cern.ch/user/p/psilva/work/Wmass/train/CMSSW_10_2_0_pre5/src/DeepML

print_help() {
    echo ""
    echo -e "${RED}Usage runRecoilRegression.sh [OPTION]... ${NC}"
    echo "Wraps up environment setting and calls to train or predict models on an input directory with ROOT trees"
    echo ""
    echo "   -r         operation to perform: prepare, train, predict (can be a CSV list)"
    echo "   -m         model number to train (see Train/test_TrainData_Recoil.py) or directory with train results for predict"
    echo "   -d         data directory (required if you start from train or predict step)"
    echo "   -p         model directory (required if you start from predict step)"
    echo "   -f         add functor outputs to the prediction (only used in the predict step)"
    echo "   -o         directory to hold the output [opt]"
    echo "   -c         cut to apply (train only)"
    echo "   -i         directory with input trees or txt file with trees (regress_results by default)"
    echo "   -t         use this tag to filter input files [opt]"
    echo "   -v         variables list index (0 bydefault)"
    echo "   -g         gpu index (if running on gpu)"
    echo ""
}

setup_env() {
    igpu=$1
    
    cd ${TRAINPATH}
    DEEPJETCORE=${TRAINPATH}/../DeepJetCore
    export PYTHONPATH="${TRAINPATH}/Train:${TRAINPATH}/modules:${DEEPJETCORE}/../:${PYTHONPATH}"
    
    #setup environment
    if [[ ${TRAINPATH} = *"CMSSW"* ]]; then
        echo "Setting up a CMSSW-based environment"
        eval `scram r -sh`
    elif [ "${igpu}" == "none" ]; then
        echo "Setting up a standalone-based environment"
        export PATH=/afs/cern.ch/user/p/psilva/work/Wmass/train/miniconda/bin:$PATH
        source activate deepjetLinux3
        export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so
    else
        echo "Using standalone-based GPU environment"
        echo "Please set up the environment manually:"
        echo "\$ export PATH=\"/path/to/miniconda3/bin:\$PATH\""
        echo "\$ source gpu_env.sh"
        export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
    fi

    export LD_LIBRARY_PATH=${DEEPJETCORE}/compiled:$LD_LIBRARY_PATH
    export PATH=${DEEPJETCORE}/bin:$PATH

    ulimit -s 65532
}

run() {

    operation=$1
    rundir=$2/${operation}
    modelTargets=${3}
    cut=$4
    varListNb=$5
    file_list=$6
    trainDataDir=$7
    modelDir=$8
    addFunctorOutput=$9

    if [ "${operation}" != "prepare" -a "${trainDataDir}" == "none" ]; then
        echo " "
        echo -e "${RED}Missing train data directory for ${operation}${NC}"
        print_help
        exit 0
    fi
    if [ "${operation}" == "predict" -a "${modelDir}" == "none" ]; then
        echo " "
        echo -e "${RED}Missing model directory for ${operation}${NC}"
        print_help
        exit 0
    fi

    mkdir -p ${rundir} 

    #prepare to define the data variables to use
    recoilSetNb=$(( varListNb / 10 ))
    recoilVarListNb=$(( varListNb % 10 )) 
    
    echo "Configuring recoil set #${recoilSetNb} and variable list #${recoilVarListNb} from varListNb=${varListNb}"

    recoilList=(tkmet npv_tkmet closest_tkmet)
    if [ "$recoilSetNb" == "1" ]; then
        recoilList=(tkmet npv_tkmet closest_tkmet puppimet invpuppimet)
    fi

    varList=""
    for i in ${recoilList[@]}; do
        varList="${varList}h${i}_pt,h${i}_phi,h${i}_n,"
        varList="${varList}h${i}_scalar_ht,h${i}_scalar_sphericity,"
        if [[ "${recoilVarListNb}" -gt "0" ]]; then
            varList="${varList}h${i}_leadpt,h${i}_leadphi,"
        fi
        if [[ "${recoilVarListNb}" -gt "1" ]]; then
            varList="${varList}h${i}_tau1,h${i}_tau2,h${i}_tau3,h${i}_tau4,h${i}_rho,"
        fi
        if [[ "${recoilVarListNb}" -gt "2" ]]; then
            varList="${varList}h${i}_thrust,h${i}_oblateness,h${i}_thrustMajor,h${i}_thrustMinor,h${i}_thrustTransverse,h${i}_thrustTransverseMinor,"
        fi
        if [[ "${recoilVarListNb}" -gt "3" ]]; then
            varList="${varList}h${i}_sphericity,h${i}_aplanarity,h${i}_C,h${i}_D,h${i}_detST,"
        fi
    done
    varList="${varList}rho,nVertF,nJetF,mindz,vx,vy,vz"
    
    #data class arguments
    modelSpecs=($(echo "${modelTargets}" | tr ":" "\n"))
    classargs="--modelMethod ${modelSpecs[0]} --target ${modelSpecs[1]} --varList ${varList}"
    if [[ "${cut}" != "none" ]]; then
        classargs="${classargs} --sel ${cut}";
    fi
    if [[ "${addFunctorOutput}" != "none" ]]; then
        classargs="${classargs} --addFunctorOutput";
    fi

    #prepare step
    if [[ "${operation}" == "prepare" ]]; then            
        echo -e "Preparing data to train for model ${RED} ${modelTargets} ${NC} with the following arguments"
        echo -e "\t ${RED} ${classargs} ${NC}"
        trainDataDir=${rundir}/data
        rm -rf ${trainDataDir}
        convertFromRoot.py -i ${file_list} -o ${trainDataDir} -c TrainData_Recoil --noRelativePaths --classArgs "${classargs}" # for testing --nforweighter 10    
    fi

    #train step
    if [[ "${operation}" == "train" ]]; then            
        echo -e "Training model"
        dcFile=${trainDataDir}/dataCollection.dc
        trainDir=${rundir}/model
        rm -rf ${trainDir}
        if [ "${igpu}" == "none" ]; then
            python Train/test_TrainData_Recoil.py ${dcFile} ${trainDir} --modelMethod ${modelTargets}
        else
            python Train/test_TrainData_Recoil.py ${dcFile} ${trainDir} --modelMethod ${modelTargets} --gpu ${igpu}
        fi
    fi
    
    if [[ "${operation}" == "predict" ]]; then    

        echo "Testing model"
        testDir=${rundir}/test
        rm -rf ${testDir}         
        convertFromRoot.py --testdatafor ${modelDir}/trainsamples.dc -i ${file_list} -o ${testDir} --noRelativePaths -c TrainData_Recoil --classArgs "${classargs}"

        echo "Running predictions"
        predictDir=${rundir}/predict
        rm -rf ${predictDir}
        predict.py ${modelDir}/KERAS_model.h5  ${testDir}/dataCollection.dc ${predictDir}
    fi
}

#parse command line
while getopts "h?r:m:d:p:fo:c:i:t:v:g:s:" opt; do
    case "$opt" in
    h|\?)
        print_help
        exit 0
        ;;
    s) TRAINPATH=$OPTARG
        ;;
    r) operation=$OPTARG
        ;;
    m) model=$OPTARG
        ;;
    d) trainDataDir=$OPTARG
       ;;
    p) modelDir=$OPTARG
        ;;
    f) addFunctorOutput="true"
        ;;
    o) output=$OPTARG
        ;;
    c) cut=$OPTARG
        ;;
    i) input=$OPTARG
        ;;
    t) tag=$OPTARG
        ;;
    v) varListNb=$OPTARG
        ;;
    g) igpu=$OPTARG
        ;;
    esac
done
if [[ -z $operation ]] || [[ -z $model ]] || [[ -z $input ]]; then
    print_help
    exit 0
fi
if [[ -z $igpu ]]; then
    igpu="none"
fi

#start working directory
work=`pwd`/regress_results
mkdir -p ${work}

#append the model string to the output
output=${output}/model${model}
output=${output//[:,]/_}
echo -e "Will run ${RED} ${operation} ${NC} in ${work} and copy to ${output}"

#build list of files to use
file_list=${work}/file_list.txt
rm ${file_list}
if [ -f ${input} ]; then
    if [[ $input == *.root ]]; then
        echo $input > ${file_list}
        cat ${file_list}
    else
        #if another format assume it's a txt file to use
        file_list=${input}
    fi
else
    a=(`ls ${input}`)
    for i in ${a[@]}; do
        file=${input}/${i};
        if [[ -n ${tag} ]]; then
            if [[ $file != *"${tag}"* ]]; then
                continue
            fi
        fi
        echo ${file} >> ${file_list}
    done
fi
echo "# input files found in ${input} : `cat ${file_list} | wc -l`"

#setup environment
setup_env ${igpu}

#run the required operation
if [[ -z $cut ]]; then
    cut="none";
fi
if [[ -z $varListNb ]]; then
    varListNb="0";
else
    output=${output}_vlist${varListNb}
    echo "Output changed to ${output}"
fi
if [[ -z $trainDataDir ]]; then
    trainDataDir="none"
fi
if [[ -z $modelDir ]]; then
    modelDir="none"
fi
if [[ -z $addFunctorOutput ]]; then
    addFunctorOutput="none"
fi

opTokens=($(echo "${operation}" | tr "," "\n"))
for op in ${opTokens[@]}; do
    run ${op} ${work} ${model} ${cut} ${varListNb} ${file_list} ${trainDataDir} ${modelDir} ${addFunctorOutput}

    runOutputDir=${work}/${op}
    if [[ "${op}" == "prepare" ]]; then
        trainDataDir=${runOutputDir}/data
    fi
    if [[ "${op}" == "train" ]]; then
        modelDir=${runOutputDir}/model
    fi

    #prepare output directories
    if [ -n "${output}" -a "${output}" != "${work}" ]; then
        
        if [[ -d "${runOutputDir}" ]]; then

            echo -e "Moving the results to ${RED} ${output}/${op} ${NC}"
            mkdir -p ${output}/${op}
            cp -rv ${runOutputDir} ${output};
            
            #update file association to new location
            if [[ "${op}" == "predict" ]]; then
                sed -i.bak s@${runOutputDir}/predict@${output}/${op}/predict@g ${output}/${op}/predict/tree_association.txt
            fi
        fi    
    fi
done
