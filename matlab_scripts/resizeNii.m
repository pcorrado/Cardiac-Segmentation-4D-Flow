function resizeNii(fileName, factor, method)
% resizeNii resizes a nifti image in x- and y-directions by a set scale
% factor to achieve a desired resolution.
    if nargin<3; method='bilinear'; end
    img = niftiread(fileName);
    if numel(factor)==1; factor = [factor];
    info = niftiinfo(fileName);
    for z=1:size(img,3)
        for t = 1:size(img,4)
            img2(:,:,z,t) = imresize(img(:,:,z,t), factor, method); %#ok<AGROW>
        end
    end
    info.ImageSize(1:2) = round(info.ImageSize(1:2).*factor);
    info.PixelDimensions(1:2) = info.PixelDimensions(1:2)./factor;
    info.Transform.T(1:2,:) = info.Transform.T(1:2,:)./factor;
    niftiwrite(img2,fileName,info);
end

