function cnnSeg2Segment(dirName)
%cnnSeg2Segment converts nifti format short axis bSSFP images and nifti
%format LV & RV segmentation to a format compatible with Medviso Segment software.
%   Input directory should contain 'sa.nii.gz' and 'seg_sa.nii.gz' files.
    if nargin<1 || (~ischar(dirName) && ~isstring(dirName))
        error("Directory not found.\n Usage',' cnnSeg2Segment(dirName);");
    end
   
    setstruct.IM = niftiread(fullfile(dirName,'sa.nii.gz'));
    setstruct.IM = single(permute(setstruct.IM,[2,1,4,3]))./single(max(setstruct.IM(:)));
    
    seg = niftiread(fullfile(dirName,'seg_sa.nii.gz'));

    for z=1:size(seg,3)
        for t=1:size(seg,4)
            BLV = bwboundaries(seg(:,:,z,t)==1);
            [~,indLV] = max(cellfun(@numel,BLV));
            if ~isempty(indLV) && indLV>0 && size(BLV{indLV},1)>2
                pt = interparc(linspace(0,1,80),BLV{indLV}(:,1),BLV{indLV}(:,2),'csape');
                setstruct.EndoY(:,t,z) = pt(:,1);
                setstruct.EndoX(:,t,z) = pt(:,2);
            else
                setstruct.EndoX(:,t,z) = nan(80,1);
                setstruct.EndoY(:,t,z) = nan(80,1);
            end
            BRV = bwboundaries(seg(:,:,z,t)==-1);
            [~,indRV] = max(cellfun(@numel,BRV));
            if ~isempty(indRV) && indRV>0 && size(BRV{indRV},1)>2
                pt = interparc(linspace(0,1,80),BRV{indRV}(:,1),BRV{indRV}(:,2),'csape');
                setstruct.RVEndoY(:,t,z) = pt(:,1);
                setstruct.RVEndoX(:,t,z) = pt(:,2);
            else
                setstruct.RVEndoX(:,t,z) = nan(80,1);
                setstruct.RVEndoY(:,t,z) = nan(80,1);
            end
        end
    end
    
    info = niftiinfo(fullfile(dirName,'sa.nii.gz'));

    setstruct.ResolutionX = info.PixelDimensions(1);
    setstruct.ResolutionY = info.PixelDimensions(2);
    setstruct.SliceThickness = info.PixelDimensions(3);
    setstruct.SliceGap = 0;
    setstruct.TIncr = info.PixelDimensions(4)./1000;
    setstruct.XSize = info.ImageSize(1);
    setstruct.YSize = info.ImageSize(2);
    setstruct.ZSize = info.ImageSize(3);
    setstruct.TSize = info.ImageSize(4);
 
    setstruct.ImagePosition = info.Transform.T(4,1:3).*[-1,-1,1];
    R = info.Transform.T(1:3,1:3)./repmat([setstruct.ResolutionX;setstruct.ResolutionY;setstruct.SliceThickness],[1,3]);
    R = R'.*repmat([-1;-1;1],[1,3]);
    setstruct.ImageOrientation = R(1:6);
    setstruct.StartSlice=1;
    setstruct.EndSlice=1;
    setstruct.CurrentTimeFrame=1;
    setstruct.CurrentSlice=1;
    setstruct.OrgXSize=setstruct.XSize;
    setstruct.OrgYSize=setstruct.YSize;
    setstruct.OrgZSize=setstruct.ZSize;
    setstruct.OrgTSize=setstruct.TSize;
    setstruct.TDelay=0;
    setstruct.TimeVector=0:setstruct.TIncr:setstruct.TIncr*(setstruct.TSize-1);
    setstruct.EchoTime=1;
    setstruct.T2preptime=NaN;
    setstruct.RepetitionTime=1;
    setstruct.InversionTime=0;
    setstruct.TriggerTime=zeros(setstruct.ZSize,setstruct.TSize);
    setstruct.FlipAngle=1;
    setstruct.AccessionNumber='X';
    setstruct.StudyUID='';
    setstruct.StudyID='';
    setstruct.NumberOfAverages=1;
    setstruct.VENC=0;
    setstruct.GEVENCSCALE=0;
    setstruct.Scanner='GE';
    setstruct.Modality='MR';
    setstruct.PathName=dirName;
    setstruct.FileName=fullfile(dirName,'cnnSeg.mat');
    setstruct.OrigFileName='';
    setstruct.PatientInfo=struct('Name','cnn2Seg',...
                  'ID','cnn2Seg',...
           'BirthDate','19010101',...
                 'Sex','M',...
                 'Age',99,...
     'AcquisitionDate','19010101',...
              'Length',100,...
              'Weight',100,...
                 'BSA',2,...
         'Institution','');
    setstruct.XMin=1;
    setstruct.YMin=1;
    setstruct.Cyclic=1;
    setstruct.Bitstored=info.BitsPerPixel;
    setstruct.Rotated=0;
    setstruct.SequenceName='';
    setstruct.SeriesDescription='';
    setstruct.SeriesNumber=7;
    setstruct.AcquisitionTime=00001;
    setstruct.DICOMImageType='';
    setstruct.HeartRate=60./(setstruct.TIncr)./(setstruct.TSize);
    setstruct.BeatTime=(setstruct.TIncr).*(setstruct.TSize);
    setstruct.Scar=[];
    setstruct.Flow=[];
    setstruct.Report=[];
    setstruct.Perfusion=[];
    setstruct.PerfusionScoring=[];
    setstruct.Strain=[];
    setstruct.StrainTagging=[];
    setstruct.Stress=[];
    setstruct.CenterX=100;
    setstruct.CenterY=100;
    setstruct.RoiCurrent=[];
    setstruct.RoiN=0;
    setstruct.Roi=struct('X',[],'Y',[],'T',[],'Z',[],'Sign',[],'Name','','LineSpec','','Area',[],'Mean',[],'StD',[],'Flow',[]);
    setstruct.EndoPinX=[];
    setstruct.EndoPinY=[];
    setstruct.EndoInterpX=[];
    setstruct.EndoInterpY=[];
    setstruct.EndoXView=zeros(1215,setstruct.TSize);
    setstruct.EndoYView=zeros(1215,setstruct.TSize);
    setstruct.EndoPinXView=[];
    setstruct.EndoPinYView=[];
    setstruct.EpiX=zeros(80,setstruct.TSize,setstruct.ZSize);
    setstruct.EpiY=zeros(80,setstruct.TSize,setstruct.ZSize);
    setstruct.EpiXView=zeros(1215,setstruct.TSize);
    setstruct.EpiYView=zeros(1215,setstruct.TSize);
    setstruct.EpiPinX=[];
    setstruct.EpiPinY=[];
    setstruct.EpiInterpX=[];
    setstruct.EpiInterpY=[];
    setstruct.EpiPinXView=[];
    setstruct.EpiPinYView=[];
    setstruct.EndoDraged=false(setstruct.TSize,setstruct.ZSize);
    setstruct.EpiDraged=false(setstruct.TSize,setstruct.ZSize);
    setstruct.RVEndoInterpX=cell(setstruct.TSize,setstruct.ZSize);
    setstruct.RVEndoInterpY=cell(setstruct.TSize,setstruct.ZSize);
    setstruct.RVEndoXView=zeros(1,setstruct.TSize);
    setstruct.RVEndoYView=zeros(1,setstruct.TSize);
    setstruct.RVEpiX=[];
    setstruct.RVEpiY=[];
    setstruct.RVEpiInterpX=[];
    setstruct.RVEpiInterpY=[];
    setstruct.RVEpiXView=NaN;
    setstruct.RVEpiYView=NaN;
    setstruct.RVEndoPinX=[];
    setstruct.RVEndoPinY=[];
    setstruct.RVEpiPinX=[];
    setstruct.RVEpiPinY=[];
    setstruct.RVEndoPinXView=[];
    setstruct.RVEndoPinYView=[];
    setstruct.RVEpiPinXView=[];
    setstruct.RVEpiPinYView=[];
    setstruct.SectorRotation=0;
    setstruct.Mmode=zeros(1,setstruct.TSize);
    setstruct.LVV=zeros(1,setstruct.TSize);
    setstruct.EPV=zeros(1,setstruct.TSize);
    setstruct.PV=zeros(1,setstruct.TSize);
    setstruct.LVM = zeros(1,setstruct.TSize);
    setstruct.ESV=0;
    setstruct.EDV=0;
    setstruct.EDT=1;
    setstruct.EST=1;
    setstruct.SV=0;
    setstruct.EF=0;
    setstruct.PFR=544.8117;
    setstruct.PER=458.6786;
    setstruct.PFRT=11;
    setstruct.PERT=2;
    setstruct.RVV=zeros(1,setstruct.TSize);
    setstruct.RVEPV=nan(1,setstruct.TSize);
    setstruct.RVM=nan(1,setstruct.TSize);
    setstruct.RVESV=1;
    setstruct.RVEDV=1;
    setstruct.RVSV=0;
    setstruct.RVEF=0;
    setstruct.SpectSpecialTag=[];
    setstruct.RVPFR=405.1299;
    setstruct.RVPFRT=12;
    setstruct.RVPER=569.7068;
    setstruct.RVPERT=2;
    setstruct.ImageType='Cine';
    setstruct.ImageViewPlane='Short-axis';
    setstruct.ImagingTechnique='MRSSFP';
    setstruct.IntensityScaling=2644;
    setstruct.IntensityOffset=0;
    setstruct.StartAnalysis=1;
    setstruct.EndAnalysis=setstruct.TSize;
    setstruct.EndoCenter=1;
    setstruct.NormalZoomState=[0.5;0.5+setstruct.XSize;0.5;setstruct.YSize+0.5];
    setstruct.MontageZoomState=[];
    setstruct.MontageRowZoomState=[];
    setstruct.MontageFitZoomState=[];
    setstruct.Measure=[];
    setstruct.RV=struct();
    setstruct.LevelSet=[];
    setstruct.OrgRes=info.PixelDimensions./[1,1,1,1000];
    setstruct.AutoLongaxis=0;
    setstruct.Longaxis=1;
    setstruct.Point=struct('X',[],'Y',[],'T',[],'Z',[]);
    setstruct.Point.Label={};
    setstruct.IntensityMapping=struct('Brightness',0.5779,'Contrast',2.2695,'Compression',[]);
    setstruct.Colormap=[];
    setstruct.View=struct('ViewPanels',1,...
        'ViewMatrix',[1 1],'ThisFrameOnly',1,...
        'CurrentPanel',1,'CurrentTheme','rv','CurrentTool','select');
    setstruct.View.ViewPanelsType={'one'};
    setstruct.View.ViewPanelsMatrix={[4 4]};
    setstruct.RotationCenter=[];
    setstruct.Fusion=[];
    setstruct.ProgramVersion='2.2R6435';
    setstruct.PapillaryIM=[];
    setstruct.MaR=[];
    setstruct.T2=[];
    setstruct.Children=[];
    setstruct.Parent=[];
    setstruct.Linked=1;
    setstruct.Overlay=[];
    setstruct.Intersection=[];
    setstruct.Comment=[];
    setstruct.AtrialScar=[];
    setstruct.SAX3=[];
    setstruct.HLA=[];
    setstruct.VLA=[];
    setstruct.GLA=[];
    setstruct.CT=[];
    setstruct.PapillaryThreshold=0;
    setstruct.ECV=[];
    setstruct.Developer=[];
    setstruct.EndoInterpXView=[];
    setstruct.EndoInterpYView=[];
    setstruct.EpiInterpXView=[];
    setstruct.EpiInterpYView=[];
    setstruct.RVEndoInterpXView=cell(setstruct.TSize,1);
    setstruct.RVEndoInterpYView=cell(setstruct.TSize,1);
    setstruct.RVEpiInterpXView=[];
    setstruct.RVEpiInterpYView=[];
    
    
    
    im = []; %#ok<NASGU>
    preview = repmat(uint8(setstruct.IM(:,:,1,1).*255),[1,1,3]); %#ok<NASGU>
    info = struct('Name','cnn2Seg',...
                  'ID','cnn2Seg',...
           'BirthDate','19010101',...
                 'Sex','M',...
                 'Age',99,...
     'AcquisitionDate','19010101',...
              'Length',100,...
              'Weight',100,...
                 'BSA',2,...
         'Institution','',...
            'NFrames',0,...
           'NumSlices',0,...
        'ResolutionX',0,...
         'ResolutionY',0,...
      'SliceThickness',0,...
            'SliceGap',0,...
               'TIncr',0,...
            'EchoTime',0,...
           'FlipAngle',0,...
     'AccessionNumber','',...
            'StudyUID','',...
             'StudyID','',...
    'NumberOfAverages',0,...
      'RepetitionTime',0,...
       'InversionTime',0,...
              'TDelay',0,...
                'VENC',0,...
             'Scanner','',...
    'ImagingTechnique','',...
           'ImageType','',...
      'ImageViewPlane','',...
    'IntensityScaling',1,...
     'IntensityOffset',0,...
        'MultiDataSet',true,...
            'Modality','MR'); %#ok<NASGU>
    save(fullfile(dirName,'cnn2Seg.mat'),'preview','info','im','setstruct');
end
