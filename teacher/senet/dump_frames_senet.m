function dump_frames_senet(imgRoot, outRoot, p)
% SENet-50 FER+ teacher (Albanie et al., 2018), per-frame probabilities for one MAV-Celeb version.
% Run once per version, with ABSOLUTE paths (the script cd's into the MatConvNet root):
%
%   dump_frames_senet('/abs/mavceleb/v1/faces', '/abs/teacher_outputs/senet/mavceleb_v1', ...
%                     '/abs/matconvnet-1.0-beta25')
%
% imgRoot : <mavceleb>/v{N}/faces   (id*/Language/video/*.jpg)
% outRoot : one <speaker>__<Language>__<video>.csv per video: filename + 8 FER+ probabilities
% p       : MatConvNet root with contrib/mcnCrossModalEmotions and senet50-ferplus.mat in
%           contrib/mcnCrossModalEmotions/data/models
% Existing CSVs are skipped (resume support).

if ~exist(outRoot,'dir'), mkdir(outRoot); end

%% ========= INIT MATCONVNET =========
cd(p); addpath(fullfile(p,'matlab')); vl_setupnn;
addpath(genpath(fullfile(p,'contrib','autonn')));
addpath(genpath(fullfile(p,'contrib','mcnExtraLayers')));
addpath(genpath(fullfile(p,'contrib','mcnDatasets')));
addpath(genpath(fullfile(p,'contrib','mcnCrossModalEmotions')));

%% ========= LOAD SENET FER+ =========
modelsDir = fullfile(p,'contrib','mcnCrossModalEmotions','data','models');
mats = {'senet50-ferplus.mat','resnet50-ferplus.mat'};
modelMat = '';
for k=1:numel(mats)
  f=fullfile(modelsDir,mats{k});
  if exist(f,'file'), modelMat=f; break; end
end
assert(~isempty(modelMat), 'Place senet50-ferplus.mat in %s', modelsDir);
fprintf('Using model: %s\n', modelMat);

S = load(modelMat); if isfield(S,'net'), S = S.net; end
net = dagnn.DagNN.loadobj(S); net.mode='test';
norm = net.meta.normalization; imSize = norm.imageSize(1:2);
meanIm = []; if isfield(norm,'averageImage'), meanIm = norm.averageImage; end

% SENet FER+ class order (kept as-is; Python side aligns by name)
classes = {'neutral','happiness','surprise','sadness','anger','disgust','fear','contempt'};

inputVar = 'input';
if ~any(strcmp({net.vars.name},'input')) && any(strcmp({net.vars.name},'data'))
    inputVar = 'data';
end

%% ========= WALK speaker / language / video_id =========
ids = dir(imgRoot); ids = ids([ids.isdir]);
ids = ids(~ismember({ids.name},{'.','..'}));
fprintf('Found %d identities under %s\n', numel(ids), imgRoot);

for ii = 1:numel(ids)
    idName = ids(ii).name;
    idDir  = fullfile(imgRoot, idName);
    Ls = dir(idDir); Ls = Ls([Ls.isdir]); Ls = Ls(~ismember({Ls.name},{'.','..'}));

    for li = 1:numel(Ls)
        lang = Ls(li).name;
        langDir = fullfile(idDir, lang);
        V = dir(langDir); V = V([V.isdir]); V = V(~ismember({V.name},{'.','..'}));

        for vi = 1:numel(V)
            video_id = V(vi).name;
            vidDir = fullfile(langDir, video_id);

            % output file mirrors POSTER V2 naming: id__lang__video.csv
            outName = sprintf('%s__%s__%s.csv', idName, lang, video_id);
            outPath = fullfile(outRoot, outName);
            if exist(outPath,'file')      % resume support
                continue;
            end

            % gather JPGs (case-insensitive FS: match once, then dedupe)
            D = dir(fullfile(vidDir, '*.jpg'));
            imgs = fullfile({D.folder}, {D.name})';
            imgs = unique(imgs);   % guard against any duplicate matches
            if isempty(imgs), continue; end

            % sort by filename (zero-padded timestamps sort correctly)
            imgs = sort(imgs);

            n = numel(imgs);
            P = zeros(n, numel(classes), 'single');
            fnames = cell(n,1);

            for fi = 1:n
                img = imgs{fi};
                [~, stem, ext] = fileparts(img);
                fnames{fi} = [stem ext];

                im = read_resize_jpeg(img, imSize);
                imN = single(im);
                if ~isempty(meanIm)
                    if isequal(size(meanIm), size(imN))
                        imN = imN - single(meanIm);
                    elseif isequal(size(meanIm),[1 1 3])
                        imN = bsxfun(@minus, imN, single(meanIm));
                    end
                end
                net.eval({inputVar, imN});
                P(fi,:) = single(get_ferplus_probs(net, numel(classes)));
            end

            % write CSV: filename, <8 prob columns>
            T = array2table(P, 'VariableNames', classes);
            T = addvars(T, fnames, 'Before', 1, 'NewVariableNames', 'filename');
            writetable(T, outPath);

            fprintf('[%d/%d] %s/%s/%s  -> %d frames\n', ...
                ii, numel(ids), idName, lang, video_id, n);
        end
    end
end

fprintf('\n=== DONE: %s ===\n', imgRoot);

end % main

%% ===== helpers =====
function im = read_resize_jpeg(fname, imSize)
  data = vl_imreadjpeg({fname}, 'Resize', imSize, 'Interpolation', 'bilinear');
  im = data{1};
  if size(im,3)==1, im = repmat(im,[1 1 3]); end
end

function probs = get_ferplus_probs(net, K)
  scores = []; is_prob = false;
  postsoft = {'prob','softmax'};
  for n=1:numel(postsoft)
    if any(strcmp({net.vars.name}, postsoft{n}))
      v = squeeze(gather(net.vars(net.getVarIndex(postsoft{n})).value));
      if numel(v)==K, scores=v; is_prob=true; break; end
    end
  end
  if isempty(scores)
    pre = {'prediction','fc','logits','scores','score'};
    for n=1:numel(pre)
      if any(strcmp({net.vars.name}, pre{n}))
        v = squeeze(gather(net.vars(net.getVarIndex(pre{n})).value));
        if numel(v)==K, scores=v; is_prob=false; break; end
      end
    end
  end
  if isempty(scores)
    for v=numel(net.vars):-1:1
      val = net.vars(v).value;
      if ~isempty(val)
        s = squeeze(gather(val));
        if isvector(s) && numel(s)==K
          scores=s; sm=sum(s(:));
          if all(s(:)>=0) && abs(sm-1)<1e-3, is_prob=true; end
          break;
        end
      end
    end
  end
  assert(~isempty(scores),'Could not locate %d-d output.', K);
  if is_prob
    probs = max(scores(:).',0); probs = probs / max(sum(probs),1e-8);
  else
    s = scores(:)-max(scores(:)); ex=exp(s); probs=(ex/sum(ex)).';
  end
end