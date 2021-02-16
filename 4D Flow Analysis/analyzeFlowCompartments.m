function data = analyzeFlowCompartments(inputDir,tShift, maskFile)

    curDir = pwd;
    if nargin<1
         fprintf('No input directory, using %s.\n', pwd);
        inputDir = pwd;
    else
         fprintf('Computing pathlines for %s.\n', inputDir);
    end

    cd(inputDir);

    if nargin<2 || isempty(tShift) || ~isnumeric(tShift)
        tShift = 0;
    end

    fprintf('Reading images...\n');
    if nargin>2 && exist(maskFile,'file')
        mask = circshift(squeeze(niftiread(maskFile)),[0,0,0,tShift]);
    elseif exist('registeredMask.nii','file')
        mask = circshift(squeeze(round(niftiread('registeredMask.nii'))),[0,0,0,tShift]);
    else
        mask = circshift(squeeze(round(niftiread('registeredMask.nii.gz'))),[0,0,0,tShift]);
    end

    if exist('VELX.nii','file')
        ext = '.nii';
    else
        ext = '.nii.gz';
    end

    info = niftiinfo(['VELX',ext]);
    vol = prod(info.PixelDimensions(1:3)/1000); % pixel volume in m^3

    lvVolume = squeeze(sum(sum(sum(mask>0,1),2),3).*vol.*1e6); % in mL

    staticMask = false;
    if all(lvVolume==lvVolume(1)); staticMask=true; end

    [~,edvTime] = max(lvVolume);
    if staticMask
        edvTime = 7-tShift;
    end

    mask = circshift(mask,[0,0,0,-(edvTime-1)]);

    vx = double(circshift(-niftiread(['VELX',ext]),[0,0,0,-(edvTime-1)]));
    vy = double(circshift(-niftiread(['VELY',ext]),[0,0,0,-(edvTime-1)]));
    vz = double(circshift(-niftiread(['VELZ',ext]),[0,0,0,-(edvTime-1)]));

    lvVolume = squeeze(sum(sum(sum(mask>0,1),2),3).*vol.*1e6); % in mL
    [~,esvTime] = min(lvVolume);
    if staticMask % Special case to handle static mask (i.e. phantom project)
       esvTime = 13;
    end
    maskF = mask(:,:,:,1:esvTime);
    vxF = vx(:,:,:,1:esvTime);
    vyF = vy(:,:,:,1:esvTime);
    vzF = vz(:,:,:,1:esvTime);
    maskR = mask(:,:,:,[1,size(mask,4):-1:esvTime]);
    vxR = -vx(:,:,:,[1,size(mask,4):-1:esvTime]);
    vyR = -vy(:,:,:,[1,size(mask,4):-1:esvTime]);
    vzR = -vz(:,:,:,[1,size(mask,4):-1:esvTime]);

    % Done reading images.

    voxelSize = info.PixelDimensions;
    voxelSize(4) = voxelSize(4)./1000; % from milliseconds to seconds

    %  Computing forward pathlines
    fpaths = computePaths(maskF, vxF, vyF, vzF, voxelSize);

    % Computing backward pathlines
    rpaths = computePaths(maskR,vxR,vyR,vzR, voxelSize);
    rpaths(:,4,:) = mod(-rpaths(:,4,:),voxelSize(4).*size(mask,4));

    paths = [flip(rpaths(2:end,:,:),1);fpaths];
    data = classifyPaths(mask, paths, fpaths, rpaths, voxelSize);
    data.paths = paths;

     direct = numel(data.direct)/(size(data.paths,3)-numel(data.errant));
     delayed = numel(data.delayed)/(size(data.paths,3)-numel(data.errant));
     retained = numel(data.retained)/(size(data.paths,3)-numel(data.errant));
     residual = numel(data.residual)/(size(data.paths,3)-numel(data.errant));
     passing = 1-numel(data.errant)/size(data.paths,3);

     fprintf('\nPercentage passing QC check: %%%3.1f\n',passing*100);
     fprintf('\nPercentage direct flow: %%%3.1f\n',direct*100);
     fprintf('Percentage delayed ejection: %%%3.1f\n',delayed*100);
     fprintf('Percentage retained inflow: %%%3.1f\n',retained*100);
     fprintf('Percentage residual volume: %%%3.1f\n',residual*100);

    cd(curDir);
end

function data = classifyPaths(mask, paths, fPaths, rPaths, voxSize)
% Classifying pathlines

    mask = cat(4,mask,mask(:,:,:,1));

    [x,y,z,t] = ndgrid( (1:size(mask,1)).*voxSize(1),...
        (1:size(mask,2)).*voxSize(2),...
        (1:size(mask,3)).*voxSize(3),...
        (0:(size(mask,4)-1)).*voxSize(4) );

    F = griddedInterpolant(x,y,z,t,double(mask),'nearest', 'nearest');

    px = reshape(paths(:,1,:),[],1);
    py = reshape(paths(:,2,:),[],1);
    pz = reshape(paths(:,3,:),[],1);
    pt = reshape(paths(:,4,:),[],1);

    nT = size(paths,1);
    nPath = size(paths,3);
    locations = reshape(F([px,py,pz,pt]), nT, nPath);

    for ii = 1:nPath
        firstIn(ii) = find(locations(:,ii)>0,1,'first'); %#ok<AGROW>
        lastIn(ii) = find(locations(:,ii)>0,1,'last'); %#ok<AGROW>
        qcCheck(ii) = (firstIn(ii)==1  || locations(firstIn(ii),ii)==1) && ...
                      ( lastIn(ii)==nT || locations(lastIn(ii),ii)==1);             %#ok<AGROW>
    end
    startsIn = firstIn==1;
    enters = firstIn~=1;
    endsIn = lastIn==nT;
    leaves = lastIn~=nT;

    data.residual = find(startsIn .* endsIn .* qcCheck);
    data.direct   = find(enters   .* leaves .* qcCheck);
    data.retained = find(enters   .* endsIn .* qcCheck);
    data.delayed  = find(startsIn .* leaves .* qcCheck);
    data.errant   = find(~qcCheck);

    data.directStartF = squeeze(fPaths(1,1:3,data.direct)./repmat(voxSize(1:3),[1,1,numel(data.direct)]));
    data.retainedStartF = squeeze(fPaths(1,1:3,data.retained)./repmat(voxSize(1:3),[1,1,numel(data.retained)]));
    data.delayedStartF = squeeze(fPaths(1,1:3,data.delayed)./repmat(voxSize(1:3),[1,1,numel(data.delayed)]));
    data.residualStartF = squeeze(fPaths(1,1:3,data.residual)./repmat(voxSize(1:3),[1,1,numel(data.residual)]));

    data.directStartR = squeeze(rPaths(1,1:3,data.direct)./repmat(voxSize(1:3),[1,1,numel(data.direct)]));
    data.retainedStartR = squeeze(rPaths(1,1:3,data.retained)./repmat(voxSize(1:3),[1,1,numel(data.retained)]));
    data.delayedStartR = squeeze(rPaths(1,1:3,data.delayed)./repmat(voxSize(1:3),[1,1,numel(data.delayed)]));
    data.residualStartR = squeeze(rPaths(1,1:3,data.residual)./repmat(voxSize(1:3),[1,1,numel(data.residual)]));
end

function paths = computePaths(mask, vx, vy, vz, voxSize)
    % Computing pathlines

    stepSizeFraction = 100;

    [x,y,z,t] = ndgrid( (1:size(vx,1)     ).*voxSize(1),...
                        (1:size(vx,2)     ).*voxSize(2),...
                        (1:size(vx,3)     ).*voxSize(3),...
                        (0:(size(vx,4)-1)).*voxSize(4) );
    tt = (0:(size(vx,4)-1)).*voxSize(4);
    nT = numel(tt);

    ind = find(mask(:,:,:,1)>0);
    nVox = numel(ind);
    h = (voxSize(4))/stepSizeFraction;   % step size

    % Setting up interpolation grids
    if (nT>1)
        fx = griddedInterpolant(x,y,z,t,vx,'linear', 'none');
        fy = griddedInterpolant(x,y,z,t,vy,'linear', 'none');
        fz = griddedInterpolant(x,y,z,t,vz,'linear', 'none');

        F = @(tq,rq) [fx([rq,tq]), fy([rq,tq]), fz([rq,tq])];
    else
        fx = griddedInterpolant(x,y,z,vx,'linear', 'none');
        fy = griddedInterpolant(x,y,z,vy,'linear', 'none');
        fz = griddedInterpolant(x,y,z,vz,'linear', 'none');

        F = @(tq,rq) [fx(rq), fy(rq), fz(rq)];
    end


    ttt = repmat(reshape(tt,1,1,[]),[nVox,1,1]);
    paths = zeros(nVox,4,nT*stepSizeFraction);

    paths(:,:,1) = [x(ind),y(ind),z(ind),t(ind)];
    for ii=1:(nT*stepSizeFraction-1)                              % calculation loop
        k1 = F( ttt(:,1,ceil(ii/stepSizeFraction)) + 0.0*h, paths(:,1:3,ii) + 0.0*h    );
        k2 = F( ttt(:,1,ceil(ii/stepSizeFraction)) + 0.5*h, paths(:,1:3,ii) + 0.5*h*k1);
        k3 = F( ttt(:,1,ceil(ii/stepSizeFraction)) + 0.5*h, paths(:,1:3,ii) + 0.5*h*k2);
        if ii<(nT*stepSizeFraction-1)
            k4 = F( ttt(:,1,ceil(ii/stepSizeFraction)) + 1.0*h, paths(:,1:3,ii) + 1.0*h*k3);
        else
            k4 = F( ttt(:,1,nT)        , paths(:,1:3,ii) + 1.0*h*k3);
        end
        paths(:,:,ii+1) = paths(:,:,ii) + [(1/6)*(k1+2*k2+2*k3+k4)*h, repmat(h,[nVox,1])];  % main equation
    end
    paths = paths(:,:,1:stepSizeFraction:end);
    % Done computing pathlines.
    paths = permute(paths,[3,2,1]);
end
