close all; clear all; clc
warning off;
addpath(genpath('ClusteringMeasure'));
addpath(genpath('utils'));
addpath(genpath('MinMaxSelection'));
addpath(genpath('funs'));
ResSavePath = 'Res/';
MaxResSavePath = 'maxRes/';

SetParas;
for dataIndex = 1
    dataName = [dataPath datasetName{dataIndex} '.mat'];
    metric = metricName{dataIndex};
    load(dataName);
    numClust = length(unique(gt));
    knn0 = 15;
    [fea] = NormalizeData(fea);    
    ResBest = zeros(1, 8);
    ResStd = zeros(1, 8);
    
    % parameters setting
    r1 = 0;
    r2 = [1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3];
    
    acc = zeros(length(r1), length(r2));
    nmi = zeros(length(r1), length(r2));
    purity = zeros(length(r1), length(r2));
    idx = 1;
    for r1Index = 1 : length(r1)
        r1Temp = r1(r1Index);
        for r2Index = 1 : length(r2)
            r2Temp = r2(r2Index);
            % Main algorithm
            fprintf('Please wait a few minutes\n');
            disp(['Dataset: ', datasetName{dataIndex}, ...
                ', --r1--: ', num2str(r1Temp), ', --r2--: ', num2str(r2Temp)]);
            tic;
            [res] = AGLMF_main(fea, numClust, knn0, metric, gt,r2Temp);
            Runtime(idx) = toc;
            disp(['runtime: ', num2str(Runtime(idx))]);
            idx = idx + 1;
            tempResBest(1, : ) = res(1, : );
            tempResStd(1, : ) = res(2, : );
            acc(r1Index, r2Index) = tempResBest(1, 7);
            nmi(r1Index, r2Index) = tempResBest(1, 4);
            purity(r1Index, r2Index) = tempResBest(1, 8);
            resFile = [ResSavePath datasetName{dataIndex}, '-ACC=', num2str(tempResBest(1, 7)), ...
                '-r1=', num2str(r1Temp), '-r2=', num2str(r2Temp), '.mat'];
            save(resFile, 'tempResBest', 'tempResStd');
            for tempIndex = 1 : 8
                if tempResBest(1, tempIndex) > ResBest(1, tempIndex)
                    ResBest(1, tempIndex) = tempResBest(1, tempIndex);
                    ResStd(1, tempIndex) = tempResStd(1, tempIndex);
                end
            end
        end
    end
    aRuntime = mean(Runtime);
    resFile2 = [MaxResSavePath datasetName{dataIndex}, '-ACC=', num2str(ResBest(1, 7)), '.mat'];
    save(resFile2, 'ResBest', 'ResStd', 'acc', 'nmi', 'purity', 'aRuntime','gt');
    resFile3 = [MaxResSavePath datasetName{dataIndex}, '.mat'];
    save(resFile3, 'ResBest', 'ResStd', 'acc', 'nmi', 'purity', 'aRuntime','gt');
end