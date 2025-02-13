% % 'CS5567 Deep Learning Final Project' 
% % face recognition transfer learning CNN
% 
% 
% % Unzip the contents of 'GTdb_crop.zip' and 'labels_gt.zip'
% % (Georgia Tech face database from http://www.anefian.com/research/face_reco.htm) 
% into a folder named 'cropped_faces' and 'labels' respectively
unzip('GTdb_crop.zip', 'cropped_faces');
unzip('labels_gt.zip', 'labels');

% read all the labels in from labels and assign them to the list all_labels
% Define the folder path
folder_path = 'labels/labels';

% Get the list of all files without extension in the folder and subfolders
all_files = dir(fullfile(folder_path, '**', '*'));

% Filter out directories from the list of files
label_files = all_files(~[all_files.isdir]);

% Initialize an empty cell array to store the labels
all_labels = {};

% Loop through each file and read the labels
for i = 1:length(label_files)
    % Construct the full file path
    file_path = fullfile(label_files(i).folder, label_files(i).name);

    % Read the content of the file as a string
    file_content = fileread(file_path);

    % Extract the last three characters as the label
    label = file_content(end-2:end);

    % Extract only the numbers from the last three characters
    label = regexp(label, '\d', 'match');

    % Combine the numeric characters into a single string
    label = strjoin(label, '');

    % Append the label to the list of all labels
    all_labels = [all_labels; string(label)];
end
% Create an ImageDatastore object 'imds' that reads image data from the 'cropped_faces' folder
% and reads in the labels for each image from the 'labels' folder
% 50 subjects, 15 images per subject for a total of 750 images
imds = imageDatastore('cropped_faces/cropped_faces', ...
    'Labels', categorical(all_labels), ...
    'FileExtensions', '.jpg'); 

%%%% If wanting to operate under the "subject-dependent" protocol
%%%% 7 images per subject for training, 3 images per subject for validation,
%%%% and 5 images per subject for testing
%
% Split the image datastore into training/validation and testing datastores
numSubjects = 50;
numImagesPerSubject = 15;
numTrainImagesPerSubject = 7;
numValidationImagesPerSubject = 3;
numTestImagesPerSubject = 5;

% Initialize empty arrays to store the indices of the training, validation, and testing datastores
trainIndices = [];
validationIndices = [];
testIndices = [];

% Set the random seed for reproducibility
rng(0);

% Loop through each subject and get the indices for training, validation, and testing images
for i = 1:numSubjects
    subjectIndices = (i - 1) * numImagesPerSubject + 1:i * numImagesPerSubject;

    % Randomly select 10 images for training and validation
    trainValIndices = randsample(subjectIndices(1:numTrainImagesPerSubject+numValidationImagesPerSubject), ...
        numTrainImagesPerSubject + numValidationImagesPerSubject);

    % Split the selected indices into training and validation sets
    trainIndices = [trainIndices, trainValIndices(1:numTrainImagesPerSubject)];
    validationIndices = [validationIndices, trainValIndices(numTrainImagesPerSubject+1:end)];

    % Add the remaining indices to the test set
    testIndices = [testIndices, subjectIndices(numTrainImagesPerSubject + numValidationImagesPerSubject + 1:end)];
end

% Create the training, validation, and testing datastores
imdsTrain = subset(imds, trainIndices);
imdsValidation = subset(imds, validationIndices);
imdsTest = subset(imds, testIndices);
%%%%
%%%%
% %%%% If wanting to operate under the "subject-independent" protocol by 
% %%%% training your network on the first 40 subjects, and using the last 
% %%%% 10 subjects as the test set. 
% % Split the image datastore into training/validation and testing datastores
% numSubjects = 50;
% numTrainSubjects = 40;
% numTestSubjects = 10;
% numImagesPerSubject = 15;
% 
% % Get the indices of the training/validation and testing datastores
% trainValIndices = 1:numTrainSubjects * numImagesPerSubject;
% testIndices = (numTrainSubjects * numImagesPerSubject + 1):numSubjects * numImagesPerSubject;
% 
% % Create the training/validation and testing datastores
% imdsTrainVal = subset(imds, trainValIndices);
% imdsTest = subset(imds, testIndices);
% 
% % Split the training/validation datastore into 70% training and 30% validation datastores
% [imdsTrain, imdsValidation] = splitEachLabel(imdsTrainVal, 0.7, 'randomized');
% %%%%%
% %%%%%
% Calculate the number of training images
numTrainImages = numel(imdsTrain.Labels);

% Display a random sample of 25 training images in a 5x5 grid
idx = randperm(numTrainImages,25);
figure
for i = 1:25 
    subplot(5,5,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

% Load the pre-trained AlexNet or VGG19 network
%net = alexnet;
net = vgg19;

% Analyze the network structure
analyzeNetwork(net)

% Get the input size of the network
inputSize = net.Layers(1).InputSize;

% Remove the last three layers of the network to prepare for transfer learning
layersTransfer = net.Layers(1:end-3);

% Calculate the number of unique classes in the training data
numClasses = numel(categories(imdsTrain.Labels));

% Define the new layers for transfer learning
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Define the parameters for image augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

% Create augmented image datastores for training images
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% automatically resize the validation and test images without performing further data augmentation
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

% Define the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network using transfer learning
%netTransfer = trainNetwork(augimdsTrain,layers,options);

% Classify the validation images using the trained network
[YPred,scores] = classify(netTransfer,augimdsValidation);

% Calculate the accuracy of the predictions by comparing them to the true labels
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

% Extract features from the fc7 layer
fc7Layer = 'fc7'; % for VGG19

featuresTrain = activations(netTransfer, augimdsTrain, fc7Layer, 'OutputAs', 'columns');
featuresTest = activations(netTransfer, augimdsTest, fc7Layer, 'OutputAs', 'columns');

% Calculate cosine similarity between enrollment and verification images
cosineSimilarity = @(x, y) dot(x, y) / (norm(x) * norm(y));
enrollmentIdx = 1:5:size(featuresTest, 2);
verificationIdx = [2:5:size(featuresTest, 2), 3:5:size(featuresTest, 2), 4:5:size(featuresTest, 2), 5:5:size(featuresTest, 2)];
scores = zeros( numel(enrollmentIdx) , numel(verificationIdx));

for i = 1:numel(enrollmentIdx)
    enrollmentFeature = featuresTest(:, enrollmentIdx(i));
    for j = 1:size(verificationIdx, 2)
        verificationFeature = featuresTest(:, verificationIdx(j));
        similarity = cosineSimilarity(enrollmentFeature, verificationFeature);
        scores(i,j) = similarity;
    end
end


% Create genuine and impostor score sets
genuineScores = diag(scores);
impostorScores = scores(:);
impostorScores = setdiff(impostorScores, genuineScores);

% Plot histograms of genuine and impostor scores
figure;
histogram(genuineScores, 'Normalization', 'pdf', 'BinWidth', 0.1, 'FaceColor', 'b');
hold on;
histogram(impostorScores, 'Normalization', 'pdf', 'BinWidth', 0.1, 'FaceColor', 'r');
xlabel('Cosine Similarity');
ylabel('Probability Density');
legend('Genuine Scores', 'Impostor Scores');
grid on;

% Compute and plot ROC curve
[FPR, TPR, T, AUC] = perfcurve([ones(size(genuineScores)); zeros(size(impostorScores))], ...
                               [genuineScores; impostorScores], 1);
figure;
plot(FPR, TPR);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC), ')']);
grid on;

% Calculate d-prime (d')
meanGenuine = mean(genuineScores);
meanImpostor = mean(impostorScores);
stdGenuine = std(genuineScores);
stdImpostor = std(impostorScores);
dPrime = (meanGenuine - meanImpostor) / sqrt(0.5 * (stdGenuine^2 + stdImpostor^2));

% Display the  d-prime
fprintf("d-prime: %f\n", dPrime);

% Generate the average rank 1 and rank 5 identification rates for your system by
% using the first image (from the subject-dependent testing subset of each identity) 
% as the probe and the rest as the gallery.
%
% Create probe set and gallery set
probeSet = featuresTest(:, 1:5:end);
gallerySet = featuresTest;
gallerySet(:, 1:5:end) = []; % Remove probe images from gallery set

% Calculate cosine similarity between probe and gallery images
probeGallerySimilarity = zeros(size(probeSet, 2), size(gallerySet, 2));
for i = 1:size(probeSet, 2)
    for j = 1:size(gallerySet, 2)
        probeGallerySimilarity(i, j) = cosineSimilarity(probeSet(:, i), gallerySet(:, j));
    end
end

% Sort the similarity scores in descending order
[sortedSimilarity, sortedIndices] = sort(probeGallerySimilarity, 2, 'descend');

% Create an array to store the subject labels of the gallery images
gallerySubjects = repelem(1:numSubjects, 4);

% Create gallerySubjects matrix with the same dimensions as sortedIndices
gallerySubjectsMatrix = repmat(gallerySubjects, size(probeSet, 2), 1);

% Use element-wise indexing to create the sortedSubjects matrix
sortedSubjects = gallerySubjectsMatrix(sub2ind(size(gallerySubjectsMatrix), repmat((1:size(probeSet, 2))', 1, size(gallerySet, 2)), sortedIndices));

% Calculate the rank-1 and rank-5 identification rates
probeSubjects = 1:50;
rank1 = sum(sortedSubjects(:, 1) == probeSubjects') / size(probeSet, 2);
rank5 = sum(any(sortedSubjects(:, 1:5) == probeSubjects', 2)) / size(probeSet, 2);

% Display the rank-1 and rank-5 identification rates
fprintf("Rank-1 identification rate: %i percent", rank1*100);
fprintf("\n");
fprintf("Rank-5 identification rate: %i percent", rank5*100);
fprintf("\n");

% Calculate and display the rank-1 and rank-5 identification rates with 
% a matching threshold, considering only top-ranked gallery images that 
% pass the threshold when compared against the probe.
% Calculate EER
FAR_FRR_diff = abs(FPR - (1-TPR));
[minDiff, idxEER] = min(FAR_FRR_diff);
EER = (FPR(idxEER) + (1-TPR(idxEER))) / 2;
threshold = T(idxEER);

% Filter the similarity scores based on the match threshold
filteredSimilarity = probeGallerySimilarity;
filteredSimilarity(filteredSimilarity < threshold) = 0;

% Sort the filtered similarity scores in descending order
[sortedFilteredSimilarity, sortedFilteredIndices] = sort(filteredSimilarity, 2, 'descend');

% Create the sortedSubjects matrix with the filtered indices
sortedFilteredSubjects = gallerySubjectsMatrix(sub2ind(size(gallerySubjectsMatrix), repmat((1:size(probeSet, 2))', 1, size(gallerySet, 2)), sortedFilteredIndices));

% Calculate the rank-1 and rank-5 identification rates for filtered results
filteredRank1 = sum(sortedFilteredSubjects(:, 1) == probeSubjects') / size(probeSet, 2);
filteredRank5 = sum(any(sortedFilteredSubjects(:, 1:5) == probeSubjects', 2)) / size(probeSet, 2);

% Display the threshold
fprintf("Matching Threshold: %f\n", threshold);

% Display the rank-1 and rank-5 identification rates for filtered results
fprintf("Filtered Rank-1 identification rate: %i percent", filteredRank1*100);
fprintf("\n");
fprintf("Filtered Rank-5 identification rate: %i percent", filteredRank5*100);
fprintf("\n");
