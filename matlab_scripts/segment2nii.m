function segment2nii(matFile)
% segment2nii converts a saved session file from Medviso Segment software
% into nifti files for the short-axis image stack and for the LV/RV
% segmentation.
    
    if nargin<1 || (~ischar(matFile) && ~isstring(matFile)) || ~exist(matFile, 'file')
        error("Segment .mat file not found.\n Usage: segment2nii('segment_file.mat');");
    end
    [path,~,~] = fileparts(matFile);

    segmentData = load(matFile);
    for ii=1:numel(segmentData.setstruct)
        if strcmp(segmentData.setstruct(ii).ImageViewPlane, 'Short-axis') && segmentData.setstruct(ii).ZSize>1
            dat = segmentData.setstruct(ii);
        end
    end
    if ~exist('dat','var'); error('Could not find short-axis dataset.'); end
    
    endoX = dat.EndoX;
    endoY = dat.EndoY;
    rvEndoX = dat.RVEndoX;
    rvEndoY = dat.RVEndoY;
    
    p = dat.ImagePosition;
    R = dat.ImageOrientation;
    R(7:9) = -cross(R(1:3),R(4:6));
    R = reshape(R, [3,3]);
    spacing = [dat.ResolutionX,dat.ResolutionY,dat.SliceThickness+dat.SliceGap];
    img = permute(dat.IM,[2,1,4,3]);
    img = int16(img.*(2^15)./max(img(:)));
    xSize = dat.XSize;
    ySize = dat.YSize;
    zSize = dat.ZSize;
    nT = dat.TSize;
    dT = dat.TIncr*1000;
   
    info.Filename = '';
    info.Filemoddate = '';
    info.Filesize = 0;
    info.Description = '';
    info.Datatype = 'int16';
    info.BitsPerPixel = 16;   
    info.SpaceUnits = 'Millimeter';
    info.AdditiveOffset = 0;
    info.MultiplicativeScaling = 0;
    info.TimeOffset = 0;
    info.SliceCode = 'Unknown';
    info.FrequencyDimension = 0;
    info.PhaseDimension = 0;
    info.SpatialDimension = 0;
    info.DisplayIntensityRange = [0 0];
    info.TransformName = 'Sform';
    info.Qfactor = 1;
    R = R'*diag([-1,-1,1]);
    p = p.*[-1,-1,1];
    
    info.Transform = affine3d([[diag(spacing)*R;p],[0;0;0;1]]);
    info.AuxiliaryFile = 'none';
    
    info.ImageSize = [ySize, xSize, zSize, nT];
    info.PixelDimensions = [spacing, dT];
    info.TimeUnits = 'Millisecond';
    
    [y,x,z,t] = ndgrid(1:ySize,1:xSize,1:zSize,1:nT);
    index = findIndexInBoundary(x(:),y(:),z(:),t(:),endoX,endoY);
    indexRV = findIndexInBoundary(x(:),y(:),z(:),t(:),rvEndoX,rvEndoY);

    [zBase,zMid,zApex] = divideSlices(t(index),z(index), info.Transform.T(3,2));
    segImg = int16(zeros(ySize,xSize,zSize,nT));

    
    for tt = 1:numel(zBase)
        ind = index(t(index)==tt);
        segImg(ind(ismember(z(ind),zBase{tt})))=int16(1);
        segImg(ind(ismember(z(ind),zMid{tt})))=int16(2);
        segImg(ind(ismember(z(ind),zApex{tt})))=int16(3);
        indRV = indexRV(t(indexRV)==tt);
        segImg(indRV(ismember(z(indRV),zBase{tt})))=int16(-1);
        segImg(indRV(ismember(z(indRV),zMid{tt})))=int16(-2);
        segImg(indRV(ismember(z(indRV),zApex{tt})))=int16(-3);
    end
    
    niftiwrite(img, fullfile(path,'saSegment.nii'), info);
    niftiwrite(segImg, fullfile(path,'saSegmentMask.nii'), info);
    
    info.ImageSize = info.ImageSize(1:3);
    info.PixelDimensions = info.PixelDimensions(1:3);
    info.TimeUnits = 'None';
    niftiwrite(int16(mean(img,4)), fullfile(path,'saSegmentAvg.nii'), info);
end

function index = findIndexInBoundary(x,y,z,t,bX,bY)
    index = [];
    for tInd = 1:size(bX,2)
        for zInd = 1:size(bX,3)
            if all(isnan(bX(:,tInd,zInd)))
                continue;
            end
            ind = find((round(t)==tInd).*round(z)==zInd);
            subInd = inpolygon(x(ind),y(ind),bX(:,tInd,zInd),bY(:,tInd,zInd));
            if ~isempty(ind)
                index = [index;ind(subInd)];    %#ok<AGROW>
            end
        end
    end
end
function [zBase,zMid,zApex] = divideSlices(t,z,ky)
    a = unique([t,z],'rows');
    for tt=min(a(:,1)):max(a(:,1))
        ind = find(a(:,1)==tt);
        zBase{tt} = min(a(ind,2)):round(min(a(ind,2)) + (max(a(ind,2))-min(a(ind,2))-2)/3);%#ok<AGROW>
        zApex{tt} = round(max(a(ind,2)) - (max(a(ind,2))-min(a(ind,2))-2)/3):max(a(ind,2)); %#ok<AGROW>
        zMid{tt} = max(zBase{tt}+1):min(zApex{tt}-1); %#ok<AGROW>
    end
    if ky<0
        temp = zBase;
        zBase = zApex;
        zApex = temp;
    end
end