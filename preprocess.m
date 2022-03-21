% Get a folder containing the images to be forward processed
folderInput = uigetdir('','Folder for the input images');
if isnumeric(folderInput); return; end % User cancelled

% Ask for the output folder
folderOutput = uigetdir('','Folder1 for the output images');
if isnumeric(folderOutput); return; end % User cancelled

%% Search for all the PNG images
allFiles = dir(fullfile(folderInput, '*.png'));

% Go through all the images
for fileID = 1:length(allFiles)
    
    % Open the image
    imageData = imread([folderInput '/' allFiles(fileID).name]);
    % Add noise
    sigma = 10;
    NoisyImage = double(imageData)+randn(size(imageData))*sigma;
    
    % Save the image as 8 bit PNG
    imwrite(uint8(NoisyImage),[folderOutput '/' allFiles(fileID).name(1:end-4) '.png']);
    
    % Show progress
    fprintf(1,'%s\n',allFiles(fileID).name);
end
