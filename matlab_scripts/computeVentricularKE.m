function data = computeVentricularKE(srcDir,tshift,maskFile)
    %computeVentricularKE   compute ventricular kinetic energy
%   data = computeVentricularKE(srcDir) calculate KE-tiem curve based on 4D
%   flow data and mask located in directory 'srcDir'
%   The directory should contain 4D flow data in nifti format (MAG.nii,
%   CD.nii, VELX.nii, VELY.nii, VELZ.nii, registeredMask.nii)


    % Parse Input 1
    curDir = pwd; 
    if nargin<1 || (~ischar(srcDir) && ~isstring(srcDir)) || ~exist(srcDir, 'dir')
        srcDir = curDir;    
        fprintf('Setting source directory to %s\n', srcDir);
    end
    srcDir = char(srcDir);
    
    % Parse Input 2
    if nargin<2
        PULSEGATED = false;
        if PULSEGATED; tshift=[4,0]; else tshift=[0,0]; end   %#ok<UNRCH,SEPEX>
    end
        
    rho = 1060; % blood density in kg/m^3
    
    % Check whether niftis are compressed or not
    if exist(fullfile(srcDir,'VELX.nii'),'file')
        ext = '.nii';
    else
        ext = '.nii.gz';
    end
    
    % Read 4D flow data
    info = niftiinfo(fullfile(srcDir,['VELX',ext]));
    velX = circshift(niftiread(fullfile(srcDir,['VELX',ext])),[0,0,0,tshift(1)]);
    velY = circshift(niftiread(fullfile(srcDir,['VELY',ext])),[0,0,0,tshift(1)]);
    velZ = circshift(niftiread(fullfile(srcDir,['VELZ',ext])),[0,0,0,tshift(1)]);
    vol = prod(info.PixelDimensions(1:3)/1000); % pixel volume in m^3
    tt = linspace(0,(info.ImageSize(4)-1)*info.PixelDimensions(4),info.ImageSize(4));

    % Read LV RV mask 
    if nargin>2 && exist(maskFile,'file')
        mask = circshift(squeeze(niftiread(maskFile)),[0,0,0,tshift(2)]);
    elseif exist(fullfile(srcDir,'registeredMask.nii'),'file')
        mask = circshift(squeeze(round(niftiread(fullfile(srcDir,'registeredMask.nii')))),[0,0,0,tshift(2)]);
    else
        mask = circshift(squeeze(round(niftiread(fullfile(srcDir,'registeredMask.nii.gz')))),[0,0,0,tshift(2)]);
    end
      
    % Loop through time and compute KE at every time point
    for t=1:size(velZ,4)
        if size(mask,4)>1
            m = mask(:,:,:,t);
        else
            m = mask;
        end

        vX = double(velX(:,:,:,t))./1000; % in m/s
        vY = double(velY(:,:,:,t))./1000; % in m/s
        vZ = double(velZ(:,:,:,t))./1000; % in m/s
        vMag = sqrt(vX.^2 + vY.^2 + vZ.^2); % in m/s
        
        rvInd = m<0;
        lvInd = m>0;
        
        % Compute KE
        data.rv.KE(t) = (1/2)*rho*vol*sum(vMag(rvInd).^2).*1e6; % in uJ
        data.lv.KE(t) = (1/2)*rho*vol*sum(vMag(lvInd).^2).*1e6; % in uJ
        
        % Compute ventricular volume
        data.lv.vol(t) = sum(lvInd(:)).*vol.*1e6; % in mL
        data.rv.vol(t) = sum(rvInd(:)).*vol.*1e6; % in mL
    end
    
    % Estimate peaks
    data.lv = analyzeVentricleKE(data.lv, tt, 'LV');
    data.rv = analyzeVentricleKE(data.rv, tt, 'RV');
    
end

% Estimate KE peaks from KE vs. time curve
function  data = analyzeVentricleKE(data, t, ventricle)
    KE = data.KE;
    vol = data.vol;
    
    dVdt = (circshift(vol,1)-circshift(vol,-1)); % Rate of change of ventricular volume
    [ESV,EST] = min(vol);
    [EDV,~] = max(vol);
    SV = EDV-ESV;

    sysInd=1:EST;
    diasInd = setdiff(1:numel(vol),sysInd);

    eWaveInd = diasInd(1:round(numel(diasInd)*7/10));
    [~,iEWave] = min(dVdt(eWaveInd));
    iEWave = eWaveInd(iEWave); % index of e-wave peak

    aWaveInd = diasInd((round(numel(diasInd)*7/10+1):end));
    [~,iAWave] = min(dVdt(aWaveInd));
    iAWave = aWaveInd(iAWave); % index of a-wave peak

    [~,iDiastasis] = min(abs(dVdt(iEWave:iAWave)));
    iDiastasis = iDiastasis+iEWave-1;
    
    eWaveInd = diasInd(1:find(diasInd==iDiastasis));
    aWaveInd = diasInd((find(diasInd==iDiastasis)+1):end);

    data.KE_EDV = KE./EDV; % in uJ/mL
    data.KE_SV = KE./SV; % in uJ/mL
    data.KE = KE./1000; % in mJ

    data.sysKE = mean(data.KE(sysInd)); % mean systolic KE
    data.sysKE_EDV = mean(data.KE_EDV(sysInd));
    data.sysKE_SV = mean(data.KE_SV(sysInd));

    data.diasKE = mean(data.KE(diasInd)); % mean diastolic KE
    data.diasKE_EDV = mean(data.KE_EDV(diasInd));
    data.diasKE_SV = mean(data.KE_SV(diasInd));
    
    data.aveKE = mean(data.KE); % average KE
    data.aveKE_EDV = mean(data.KE_EDV);
    data.aveKE_SV = mean(data.KE_SV);

    [data.minKE,imin] = min(data.KE); % minimum KE
    data.minKE_EDV = min(data.KE_EDV);
    data.minKE_SV = min(data.KE_SV);

    [data.maxSysKE,imaxS] = max(data.KE(sysInd));% peak systolic KE
    imaxS = sysInd(imaxS);
    data.maxSysKE_EDV = max(data.KE_EDV(sysInd));
    data.maxSysKE_SV = max(data.KE_SV(sysInd));

    [data.maxEWKE,imaxE] = max(data.KE(eWaveInd));% E-wave KE
    imaxE = eWaveInd(imaxE);
    data.maxEWKE_EDV = max(data.KE_EDV(eWaveInd));
    data.maxEWKE_SV = max(data.KE_SV(eWaveInd));
    
    [data.maxAWKE,imaxA] = max(data.KE(aWaveInd)); % A-Wave KE
    imaxA = aWaveInd(imaxA);
    data.maxAWKE_EDV = max(data.KE_EDV(aWaveInd));
    data.maxAWKE_SV = max(data.KE_SV(aWaveInd));
    
    % Plot ventricular volume vs. time curve
    figure();
    plot(t,vol);
    ax=gca;
    yl = ax.YLim;
    hold on;
    line([t(iDiastasis),t(iDiastasis)],[0,yl(2)],'LineStyle','--');
    hold off;
    ylim([0,yl(2)]);
    title(sprintf('%s Volume vs. Time Curve',ventricle));
    xlabel('Time (ms)');
    ylabel('Volume (mL)');
    legend('Volume','Diastasis');
    
    % Plot KE-time curve with peaks
    figure();
    plot(t,data.KE_EDV,'-r*');
    ax=gca;
    yl = ax.YLim;
    hold on;
    line([t(imaxS),t(imaxS)],[0,yl(2)],'Color','blue','LineStyle','--');
    line([t(imin),t(imin)],[0,yl(2)],'Color','red','LineStyle','--');
    line([t(imaxE),t(imaxE)],[0,yl(2)],'Color','green','LineStyle','--');
    line([t(imaxA),t(imaxA)],[0,yl(2)],'Color','magenta','LineStyle','--');
    hold off;
    ylim([0,yl(2)]);
    title(sprintf('%s Kinetic Energy Time Curve',ventricle));
    xlabel('Time (ms)');
    ylabel('Kinetic Energy/EDV (uJ/mL)');
    legend(sprintf('%s KE-Time Curve',ventricle),'max systolic KE','minimum KE','max E-Wave Ke','max A-wave KE');
end