function Iout = readAndPreproc( inFilename )
    % Read image
    I = imread(inFilename);
    
    % in the case of grayscale, catenate the image to create 3-dimension image 
    if ismatrix(I)         
        I = cat(3,I,I,I);
    end
    % resize the image to 227*227*3
    Iout = imresize(I, [227 227]);  
end

%% Copyright 2017-2020 The MathWorks, Inc.