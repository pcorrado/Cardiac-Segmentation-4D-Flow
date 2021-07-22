function pcvipr2nii(srcDir)
%pcvipr2nii converts PC VIPR 4D flow data (*.dat format) to nifti format

    curDir = pwd; 
    if nargin<1 || (~ischar(srcDir) && ~isstring(srcDir)) || ~exist(srcDir, 'dir')
        srcDir = curDir;    
        fprintf('Setting source directory to %s\n', srcDir);
    end
    srcDir = char(srcDir);
    
    pcVIPRHeaderFilePath = fullfile(srcDir,'pcvipr_header.txt');
    if exist(pcVIPRHeaderFilePath, 'file')==2
        fid = fopen(pcVIPRHeaderFilePath);
        if fid<0
            error('Could not open pcvipr_header.txt file.');
        else
            C = textscan(fid,'%s %s');
            field = C{1};
            value = C{2};
            fclose(fid);
            
            fov = lookup(field,value,'fovx',3);
            
            xSize = lookup(field,value,'matrixx',1);
            ySize = lookup(field,value,'matrixy',1);
            zSize = lookup(field,value,'matrixz',1);
            nT = lookup(field,value,'frames',1);
            dT = lookup(field,value,'timeres',1);
            
            p = lookup(field,value,'sx',3)';
            R = reshape(lookup(field,value,'ix',9),[3,3])';
            spacing = sqrt(diag(R*R'));
            
            cd = zeros(xSize,ySize,zSize,nT, 'int16');
            mag = zeros(xSize,ySize,zSize,nT, 'int16');
            velX = zeros(xSize,ySize,zSize,nT, 'int16');
            velY = zeros(xSize,ySize,zSize,nT, 'int16');
            velZ = zeros(xSize,ySize,zSize,nT, 'int16');
            avgCd = zeros(xSize,ySize,zSize, 'int16'); %#ok<PREALL>
            avgMag = zeros(xSize,ySize,zSize, 'int16'); %#ok<PREALL>
            avgVelX = zeros(xSize,ySize,zSize, 'int16'); %#ok<PREALL>
            avgVelY = zeros(xSize,ySize,zSize, 'int16'); %#ok<PREALL>
            avgVelZ = zeros(xSize,ySize,zSize, 'int16'); %#ok<PREALL>
            
            name = fullfile(srcDir,'CD.dat');
            if ~exist(name, 'file')
                error('Could not find CD.dat file.');
            end
            m = memmapfile(name,'Format','int16');
            avgCd = reshape(m.Data,[xSize,ySize,zSize]);
            
            name = fullfile(srcDir,'MAG.dat');
            if ~exist(name, 'file')
                error('Could not find MAG.dat file.');
            end
            m = memmapfile(name,'Format','int16');
            avgMag = reshape(m.Data,[xSize,ySize,zSize]);
            
            name = fullfile(srcDir,'comp_vd_1.dat');
            if ~exist(name, 'file')
                error('Could not find comp_vd_1.dat file.');
            end
            m = memmapfile(name,'Format','int16');
            avgVelX = reshape(m.Data,[xSize,ySize,zSize]);
            
            name = fullfile(srcDir,'comp_vd_2.dat');
            if ~exist(name, 'file')
                error('Could not find comp_vd_2.dat file.');
            end
            m = memmapfile(name,'Format','int16');
            avgVelY = reshape(m.Data,[xSize,ySize,zSize]);
            
            name = fullfile(srcDir,'comp_vd_3.dat');
            if ~exist(name, 'file')
                error('Could not find comp_vd_3.dat file.');
            end
            m = memmapfile(name,'Format','int16');
            avgVelZ = reshape(m.Data,[xSize,ySize,zSize]);
            
            
            if nT==1
                mag = avgMag; cd = avgCd; velX = avgVelX; velY = avgVelY; velZ = avgVelZ;
            else
                for ii=1:nT
                    name = fullfile(srcDir,sprintf('ph_%03d_cd.dat',ii-1));
                    if ~exist(name, 'file')
                        error('Could not find ph_%03d_cd.dat file.',ii-1);
                    end
                    m = memmapfile(name,'Format','int16');
                    cd(:,:,:,ii) = reshape(m.Data,[xSize,ySize,zSize]);

                    name = fullfile(srcDir,sprintf('ph_%03d_mag.dat',ii-1));
                    if ~exist(name, 'file')
                        error('Could not find ph_%03d_mag.dat file.',ii-1);
                    end
                    m = memmapfile(name,'Format','int16');
                    mag(:,:,:,ii) = reshape(m.Data,[xSize,ySize,zSize]);

                    name = fullfile(srcDir,sprintf('ph_%03d_vd_1.dat',ii-1));
                    if ~exist(name, 'file')
                        error('Could not find file "ph_%03d_vd_1.dat".',ii-1);
                    end
                    m = memmapfile(name,'Format','int16');
                    velX(:,:,:,ii) = reshape(m.Data,[xSize,ySize,zSize]);

                    name = fullfile(srcDir,sprintf('ph_%03d_vd_2.dat',ii-1));

                    if ~exist(name, 'file')
                        error('Could not find file "ph_%03d_vd_2.dat".',ii-1);
                    end
                    m = memmapfile(name,'Format','int16');
                    velY(:,:,:,ii) = reshape(m.data,[xSize,ySize,zSize]);

                    name = fullfile(srcDir,sprintf('ph_%03d_vd_3.dat',ii-1));
                    if ~exist(name, 'file')
                        error('Could not find file "ph_%03d_vd_3.dat".',ii-1);
                    end
                    m = memmapfile(name,'Format','int16');
                    velZ(:,:,:,ii) = reshape(m.Data,[xSize,ySize,zSize]);
                end
            end
        end
    else
        error('Could not find pcvipr_header.txt file.');
    end
    
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
    R = R.*repmat([-1,-1,1],3,1);
    p = p.*[-1,-1,1];
%     if det(R)>0
        info.Qfactor = 1;
%     else
%         info.Qfactor = -1;
%         R(:,3) = -R(:,3);
%     end
    info.Transform = affine3d([[R;p],[0;0;0;1]]);
    info.AuxiliaryFile = 'none';
    
    if nT==1
        info.ImageSize = [xSize, ySize, zSize];
        info.PixelDimensions = spacing';
        info.TimeUnits = 'None';
        niftiwrite(mag, fullfile(srcDir,"MAG.nii"), info);
        niftiwrite(cd, fullfile(srcDir,"CD.nii"), info);
        niftiwrite(velX, fullfile(srcDir,"VELX.nii"), info);
        niftiwrite(velY, fullfile(srcDir,"VELY.nii"), info);
        niftiwrite(velZ, fullfile(srcDir,"VELZ.nii"), info);

    elseif nT>1
        info.ImageSize = [xSize, ySize, zSize, nT];
        info.PixelDimensions = [spacing', dT];
        info.TimeUnits = 'Millisecond';
        niftiwrite(mag, fullfile(srcDir,"MAG.nii"), info);
        niftiwrite(cd, fullfile(srcDir,"CD.nii)"), info);
        niftiwrite(velX, fullfile(srcDir,"VELX.nii"), info);
        niftiwrite(velY, fullfile(srcDir,"VELY.nii"), info);
        niftiwrite(velZ, fullfile(srcDir,"VELZ.nii"), info);

        info.ImageSize = info.ImageSize(1:3);
        info.PixelDimensions = info.PixelDimensions(1:3);
        info.raw.dim(1) = 3;
        info.raw.dim(5) = 1;
        info.TimeUnits = 'None';
        niftiwrite(avgMag, fullfile(srcDir,"AVG_MAG.nii"), info);
        niftiwrite(avgCd, fullfile(srcDir,"AVG_CD.nii"), info);
        niftiwrite(avgVelX, fullfile(srcDir,"AVG_VELX.nii"), info);
        niftiwrite(avgVelY, fullfile(srcDir,"AVG_VELY.nii"), info);
        niftiwrite(avgVelZ, fullfile(srcDir,"AVG_VELZ.nii"), info);
    end
end

function value = lookup(fields,values,field, length)
    index = find(cellfun(@(s) strcmp(field, s), fields));
    value = cellfun(@str2num,values(index:(index+length-1)));
end