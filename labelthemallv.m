function labelthemallv3
%% =================== CONFIG (MAV-Celeb v3, video-disjoint) ===================
% MatConvNet root
p = 'D:\albanie\matconvnet-1.0-beta25';


imgRoot = 'link image root ';
wavRoot = 'link audio root';


splitTag   = 'test';  % fill in train/val/test
splitDir = 'link to split';
videoKeyFile = fullfile(splitDir, sprintf('video_keys_%s.txt', splitTag));  % accepts "id lang video" OR "id/lang/video"


outRoot = fullfile(p, 'ferplus_out_v3_new', splitTag); %where output is saved
if ~exist(outRoot,'dir'), mkdir(outRoot); end


frameStride = 1;         
maxFramesPerVideo = Inf;  


min_frames = 20;           % require at least this many face frames per segment
max_neutral_share = 0.95;  % drop segments >95% neutral (set Inf to disable)


cd(p); addpath(fullfile(p,'matlab')); vl_setupnn;
addpath(genpath(fullfile(p,'contrib','autonn')));
addpath(genpath(fullfile(p,'contrib','mcnExtraLayers')));
addpath(genpath(fullfile(p,'contrib','mcnDatasets')));
addpath(genpath(fullfile(p,'contrib','mcnCrossModalEmotions')));


modelsDir = fullfile(p,'contrib','mcnCrossModalEmotions','data','models');
mats = {'senet50-ferplus.mat','resnet50-ferplus.mat'};
modelMat = '';
for k=1:numel(mats)
  f=fullfile(modelsDir,mats{k});
  if exist(f,'file'), modelMat=f; break; end
end
assert(~isempty(modelMat), 'Place senet50-ferplus.mat or resnet50-ferplus.mat in %s', modelsDir);
fprintf('Using model: %s\n', modelMat);

S = load(modelMat); if isfield(S,'net'), S = S.net; end
net = dagnn.DagNN.loadobj(S); net.mode='test';
norm = net.meta.normalization; imSize = norm.imageSize(1:2);
meanIm = []; if isfield(norm,'averageImage'), meanIm = norm.averageImage; end
classes = {'neutral','happiness','surprise','sadness','anger','disgust','fear','contempt'};


inputVar = 'input';
if ~any(strcmp({net.vars.name}, 'input')) && any(strcmp({net.vars.name}, 'data'))
    inputVar = 'data';
end


assert(exist(videoKeyFile,'file')==2, 'Video-key file not found: %s', videoKeyFile);
allowSet = load_allowed_video_keys(videoKeyFile);  % containers.Map of "id|lang|video" -> true


ids = dir(imgRoot);
ids = ids([ids.isdir]);
ids = ids(~ismember({ids.name},{'.','..'}));
fprintf('Found %d identities under %s\n', numel(ids), imgRoot);
fprintf('Split file: %s\n', videoKeyFile);


framesAll = table(); segmentsAll = table(); audioAll = table();


for ii = 1:numel(ids)
    idName = ids(ii).name;
    baseDir = fullfile(imgRoot, idName);
    fprintf('\n=== [%d/%d] %s ===\n', ii, numel(ids), idName);

    
    D1 = dir(fullfile(baseDir,'**','*.jpg'));
    D2 = dir(fullfile(baseDir,'**','*.jpeg'));
    allFiles = [ fullfile({D1.folder},{D1.name}), fullfile({D2.folder},{D2.name}) ]';
    if isempty(allFiles)
        fprintf('No JPG/JPEG under %s -> skipping (no frames)\n', baseDir);
        continue;
    end

    
    keepMask = false(numel(allFiles),1);
    langsF   = strings(numel(allFiles),1);
    vidsF    = strings(numel(allFiles),1);
    for i=1:numel(allFiles)
        [li, vi] = parse_lang_vid(allFiles{i}, baseDir);
        li = canon_lang(li);
        langsF(i) = li;
        vidsF(i)  = vi;
        key = lower(strcat(string(idName), "|", li, "|", vi));
        keepMask(i) = isKey(allowSet, char(key));
    end
    files = allFiles(keepMask);
    langsF = langsF(keepMask);
    vidsF  = vidsF(keepMask);

    if isempty(files)
        fprintf('No allowed frames for %s (according to %s)\n', idName, videoKeyFile);
        continue;
    end

    
    files  = files(1:frameStride:end);
    langsF = langsF(1:frameStride:end);
    vidsF  = vidsF(1:frameStride:end);

    
    if isfinite(maxFramesPerVideo)
        keep = false(numel(files),1);
        seen = containers.Map('KeyType','char','ValueType','int32');
        for i=1:numel(files)
            key = sprintf('%s|%s', langsF(i), vidsF(i));
            if ~isKey(seen, key), seen(key)=0; end
            if seen(key) < maxFramesPerVideo
                keep(i) = true; seen(key) = seen(key)+1;
            end
        end
        files  = files(keep);
        langsF = langsF(keep);
        vidsF  = vidsF(keep);
    end
    fprintf('Frames to process: %d\n', numel(files));

    
    rows = {}; rowsN = 0;

    for i=1:numel(files)
        img = files{i};
        celebrity = idName;
        language  = langsF(i);
        video_id  = vidsF(i);

        
        im = read_resize_jpeg(img, imSize);  
        imN = single(im);
        if ~isempty(meanIm)
            if isequal(size(meanIm), size(imN))
                imN = imN - single(meanIm);
            elseif isequal(size(meanIm), [1 1 3])
                imN = bsxfun(@minus, imN, single(meanIm));
            end
        end

        
        try
            net.eval({inputVar, imN});
        catch ME
            error('Forward failed on %s: %s', img, ME.message);
        end

        
        scores = []; is_prob = false;

        postsoft = {'prob','softmax'};
        for n = 1:numel(postsoft)
            if any(strcmp({net.vars.name}, postsoft{n}))
                v = squeeze(gather(net.vars(net.getVarIndex(postsoft{n})).value));
                if numel(v)==8
                    scores = v; is_prob = true; break;
                end
            end
        end
        
        if isempty(scores)
            prelogits = {'prediction','fc','logits','scores','score'};
            for n = 1:numel(prelogits)
                if any(strcmp({net.vars.name}, prelogits{n}))
                    v = squeeze(gather(net.vars(net.getVarIndex(prelogits{n})).value));
                    if numel(v)==8
                        scores = v; is_prob = false; break;
                    end
                end
            end
        end
        
        if isempty(scores)
            for v = numel(net.vars):-1:1
                val = net.vars(v).value;
                if ~isempty(val)
                    s = squeeze(gather(val));
                    if isvector(s) && numel(s)==8
                        scores = s;
                        sm = sum(s(:));
                        if all(s(:) >= 0) && abs(sm - 1) < 1e-3, is_prob = true; end
                        break;
                    end
                end
            end
        end
        assert(~isempty(scores), 'Could not locate 8-d output for %s', img);

        
        if is_prob
            probs = scores(:).';
            probs = max(probs, 0);
            probs = probs / max(sum(probs), 1e-8);
        else
            s = scores(:) - max(scores(:));
            ex = exp(s); probs = (ex/sum(ex)).';
        end

        [~, idx] = max(probs);

        rowsN = rowsN + 1;
        rows(rowsN,:) = [{celebrity, language, video_id, img}, num2cell(probs), classes(idx)]; %#ok<SAGROW>

        if mod(i,500)==0, fprintf('  [%d/%d] %s\n', i, numel(files), img); end
    end

    headers = [{'celebrity','language','video_id','img_path'}, classes, {'pred_class'}];
    T = cell2table(rows, 'VariableNames', headers);

    
    outIdDir = fullfile(outRoot, 'per_identity', idName);
    if ~exist(outIdDir,'dir'), mkdir(outIdDir); end
    framesCsv = fullfile(outIdDir, sprintf('frames_%s.csv', idName));
    writetable(T, framesCsv);
    fprintf('Frames CSV -> %s (N=%d)\n', framesCsv, height(T));

    
    [grp, ~, gi] = unique(T(:,{'celebrity','language','video_id'}), 'rows');
    P = T{:, classes};
    K = max(gi);
    agg = cell(K, 4 + numel(classes) + 1);
    for k=1:K
        idx = (gi==k); Pk = P(idx,:);
        soft = max(Pk, [], 1);         
        soft = soft / sum(soft);       
        [~, idc] = max(soft);
        agg(k,:) = [ table2cell(grp(k,:)), {sum(idx)}, num2cell(soft), classes(idc) ];
    end
    headers2 = [{'celebrity','language','video_id','n_frames'}, classes, {'pred_class'}];
    Sseg = cell2table(agg, 'VariableNames', headers2);

    
    wavBaseIdent = fullfile(wavRoot, idName);
    Dwav = dir(fullfile(wavBaseIdent,'**','*.wav'));
    wav_paths = fullfile({Dwav.folder},{Dwav.name})';
    if isempty(wav_paths)
        warning('No WAVs under %s (identity skipped for audio join).', wavBaseIdent);
        framesAll   = [framesAll;   T];            
        segmentsAll = [segmentsAll; Sseg];         
        continue;
    end
    lang = strings(numel(wav_paths),1); vid = strings(numel(wav_paths),1);
    for i=1:numel(wav_paths)
        [lang(i), vid(i)] = parse_lang_vid(wav_paths{i}, wavBaseIdent); %#ok<AGROW>
    end
    lang = canon_lang(lower(lang));
    W = table(wav_paths, lang, vid, 'VariableNames', {'wav_path','language','video_id'});

   
    wavKeys = lower(strcat(string(idName), "|", string(W.language), "|", string(W.video_id)));
    isAllowedW = false(height(W),1);
    for i=1:height(W)
        isAllowedW(i) = isKey(allowSet, char(wavKeys(i)));
    end
    W = W(isAllowedW,:);

    
    keep = (Sseg.n_frames >= min_frames) & ~(Sseg.neutral > max_neutral_share);
    Suse = Sseg(keep, :);
    Suse.language = canon_lang(lower(string(Suse.language)));
    Suse.video_id = string(Suse.video_id);

    
    J = innerjoin(W, Suse, 'Keys', {'language','video_id'});
    audioCsv = fullfile(outIdDir, sprintf('audio_pseudolabels_%s.csv', idName));
    audioCols = [{'wav_path','celebrity','language','video_id','n_frames'}, classes, {'pred_class'}];
    if ~isempty(J)
        J.celebrity = repmat({idName}, height(J), 1); % add identity
        J = J(:, audioCols); % reorder
        writetable(J, audioCsv);
        fprintf('Audio pseudo-labels -> %s (N=%d)\n', audioCsv, height(J));
        audioAll = [audioAll; J]; %#ok<AGROW>
    else
        warning('No WAV matches for %s (after quality + allow filters).', idName);
    end

    
    framesAll   = [framesAll;   T];            %#ok<AGROW>
    segmentsAll = [segmentsAll; Sseg];         %#ok<AGROW>
end


framesAllCsv   = fullfile(outRoot, 'frames_predictions_all.csv');
segmentsAllCsv = fullfile(outRoot, 'segments_maxpool_all.csv');
audioAllCsv    = fullfile(outRoot, 'audio_pseudolabels_all.csv');
if ~isempty(framesAll),   writetable(framesAll,   framesAllCsv);   end
if ~isempty(segmentsAll), writetable(segmentsAll, segmentsAllCsv); end
if ~isempty(audioAll),    writetable(audioAll,    audioAllCsv);    end

fprintf('\n=== DONE (%s) ===\n', upper(splitTag));
if ~isempty(framesAll),   fprintf('Frames   : %s (N=%d)\n', framesAllCsv,   height(framesAll));   end
if ~isempty(segmentsAll), fprintf('Segments : %s (K=%d)\n', segmentsAllCsv, height(segmentsAll)); end
if ~isempty(audioAll),    fprintf('Audio    : %s (N=%d)\n',  audioAllCsv,    height(audioAll));    end


if ~isempty(audioAll)
    langsDump = {'english','deutsch'}; %fill in wanted languages
    for li = 1:numel(langsDump)
        L = langsDump{li};
        Lrows = audioAll(strcmpi(audioAll.language, L), :);
        if ~isempty(Lrows)
            outCsvL = fullfile(outRoot, sprintf('audio_pseudolabels_%s_%s.csv', splitTag, L));
            writetable(Lrows, outCsvL);
            fprintf('Per-language CSV -> %s (N=%d)\n', outCsvL, height(Lrows));
        end
    end

    
    if strcmpi(splitTag,'train')
        heardValFrac = 0.15;  

        EN = audioAll(strcmpi(audioAll.language,'english'), :);
        if ~isempty(EN)
            [EN_train, EN_heardval] = split_heardval_lang(EN, heardValFrac);
            writetable(EN_train,    fullfile(outRoot, 'en_train.csv'));
            writetable(EN_heardval, fullfile(outRoot, 'en_heardval.csv'));
            fprintf('EN train/heardval -> %d / %d rows\n', height(EN_train), height(EN_heardval));
        else
            warning('No ENGLISH rows found in audioAll for TRAIN.');
        end

        DE = audioAll(strcmpi(audioAll.language,'german'), :);
        if ~isempty(DE)
            [DE_train, DE_heardval] = split_heardval_lang(DE, heardValFrac);
            writetable(DE_train,    fullfile(outRoot, 'de_train.csv'));
            writetable(DE_heardval, fullfile(outRoot, 'de_heardval.csv'));
            fprintf('DE train/heardval -> %d / %d rows\n', height(DE_train), height(DE_heardval));
        else
            warning('No GERMAN rows found in audioAll for TRAIN.');
        end
    end

    
    if any(strcmpi(splitTag, {'val','test'}))
        ENu = audioAll(strcmpi(audioAll.language,'english'), :);
        DEu = audioAll(strcmpi(audioAll.language,'german'),  :);
        if ~isempty(ENu)
            outEN = fullfile(outRoot, sprintf('en_unheard_%s.csv', lower(splitTag)));
            writetable(ENu, outEN);
            fprintf('EN-Unheard (%s) -> %d rows\n', splitTag, height(ENu));
        end
        if ~isempty(DEu)
            outDE = fullfile(outRoot, sprintf('de_unheard_%s.csv', lower(splitTag)));
            writetable(DEu, outDE);
            fprintf('DE-Unheard (%s) -> %d rows\n', splitTag, height(DEu));
        end
    end
end



try
  statsDir = fullfile('D:\German-EnlishMav\MavCeleb_v3\dataset','stats');
  if ~exist(statsDir,'dir'), mkdir(statsDir); end

  keyFiles = struct('Train', fullfile(splitDir,'video_keys_train.txt'), ...
                    'Val',   fullfile(splitDir,'video_keys_val.txt'), ...
                    'Test',  fullfile(splitDir,'video_keys_test.txt'));
  splits = fieldnames(keyFiles);

  rows = cell(0,5);
  totSeg = 0; totSecs = 0;
  totLang = struct('german',0,'english',0);

  for si = 1:numel(splits)
      sn = splits{si};
      kf = keyFiles.(sn);
      if exist(kf,'file')~=2
          warning('Video-key list missing: %s (skipping in summary)', kf);
          continue
      end
      [idsK, langsK, vidsK] = read_video_keys(kf);

      segs = 0; secs = 0;
      byLang = struct('german',0,'english',0);

      for jj = 1:numel(idsK)
          ident = idsK(jj);
          base  = fullfile(wavRoot, ident);
          wavP  = fullfile(base, langsK(jj), vidsK(jj), '*.wav');
          Dw = dir(wavP);
          for kk = 1:numel(Dw)
              fn = fullfile(Dw(kk).folder, Dw(kk).name);
              L = canon_lang_from_path(fn);
              try
                  info = audioinfo(fn);
                  secs = secs + info.Duration;
              catch
              end
              segs = segs + 1;
              if isfield(byLang,L), byLang.(L) = byLang.(L) + 1; end
          end
      end

      rows(end+1,:) = {sn, segs, secs/3600, byLang.german, byLang.english}; %#ok<SAGROW>
      totSeg = totSeg + segs; totSecs = totSecs + secs/3600;
      totLang.german  = totLang.german  + byLang.german;
      totLang.english = totLang.english + byLang.english;
  end

  
  rows(end+1,:) = {'Total', totSeg, totSecs, totLang.german, totLang.english};

  
  fprintf('\n=== MAV-Celeb v3 by split (Segments & Hours, video-disjoint) ===\n');
  for r=1:size(rows,1)
      fprintf('%-5s  Clips=%5d  Hours=%6.2f  German=%5d  English=%5d\n', ...
          rows{r,1}, rows{r,2}, rows{r,3}, rows{r,4}, rows{r,5});
  end

  
  tex = {
    '\begin{table}[htbp]'
    '\centering'
    '\caption{MAV-Celeb v3 (unlabeled) summary by split (video-disjoint).}'
    '\label{tab:v3_unlabeled_stats}'
    '\begin{tabular}{lrrrr}'
    '\hline'
    '\textbf{Split} & \textbf{Clips} & \textbf{Hours} & \textbf{German} & \textbf{English} \\'
    '\hline'
  };
  for r=1:size(rows,1)
      split = rows{r,1}; segs = rows{r,2}; hours = rows{r,3};
      de = rows{r,4}; en = rows{r,5};
      if strcmpi(split,'Total')
          line = sprintf('\\textbf{%s} & \\textbf{%s} & \\textbf{%.2f} & \\textbf{%s} & \\textbf{%s} \\\\',...
              split, fmt_int(segs), hours, fmt_int(de), fmt_int(en));
      else
          line = sprintf('%s & %s & %.2f & %s & %s \\\\',...
              split, fmt_int(segs), hours, fmt_int(de), fmt_int(en));
      end
      tex{end+1,1} = line; 
  end
  tex(end+1,1) = {'\hline'};
  tex(end+1,1) = {'\end{tabular}'};
  tex(end+1,1) = {'\end{table}'};

  texPath = fullfile(statsDir,'v3_by_split_segments.tex');
  fid = fopen(texPath,'w'); fprintf(fid,'%s\n', tex{:}); fclose(fid);
  fprintf('LaTeX written -> %s\n', texPath);

catch ME
  warning('Summary/LaTeX block failed: %s', ME.message);
end

end 




function allowSet = load_allowed_video_keys(videoKeyFile)
  L = strtrim(string(splitlines(fileread(videoKeyFile))));
  L = L(L~="");                         
  L = L(~startsWith(L,"#"));           

  n = numel(L);
  ids   = strings(n,1);
  langs = strings(n,1);
  vids  = strings(n,1);

  for i=1:n
      li = L(i);
      parts = split(li);
      if numel(parts) >= 3
          ids(i)   = parts(1);
          langs(i) = parts(2);
          vids(i)  = parts(3);
      else
          parts2 = regexp(li, '[/\\]+', 'split');
          if numel(parts2) >= 3
              ids(i)   = parts2{1};
              langs(i) = parts2{2};
              vids(i)  = parts2{3};
          else
              ids(i) = ""; langs(i) = ""; vids(i) = "";
          end
      end
  end

  langs = canon_lang(langs);
  keep = ~(ids=="" | langs=="" | vids=="");
  ids = ids(keep); langs = langs(keep); vids = vids(keep);

  keys = lower(strcat(ids, "|", langs, "|", vids));
  keys = unique(keys);
  fprintf('Loaded %d allowed video keys from %s\n', numel(keys), videoKeyFile);
  allowSet = containers.Map(cellstr(keys), num2cell(true(size(keys))));
end

function [ids, langs, vids] = read_video_keys(videoKeyFile)
  L = strtrim(string(splitlines(fileread(videoKeyFile))));
  L = L(L~="");
  L = L(~startsWith(L,"#"));
  n = numel(L);
  ids   = strings(n,1);
  langs = strings(n,1);
  vids  = strings(n,1);
  for i=1:n
      li = L(i);
      parts = split(li);
      if numel(parts) >= 3
          ids(i)   = parts(1);
          langs(i) = parts(2);
          vids(i)  = parts(3);
      else
          parts2 = regexp(li, '[/\\]+', 'split');
          if numel(parts2) >= 3
              ids(i)   = parts2{1};
              langs(i) = parts2{2};
              vids(i)  = parts2{3};
          else
              ids(i) = ""; langs(i) = ""; vids(i) = "";
          end
      end
  end
  langs = canon_lang(langs);
  keep = ~(ids=="" | langs=="" | vids=="");
  ids = ids(keep); langs = langs(keep); vids = vids(keep);
end

function y = canon_lang(x)
  if isstring(x) || ischar(x)
      x = string(x);
  elseif iscellstr(x)
      x = string(x);
  end
  y = strings(size(x));
  for i=1:numel(x)
      t = lower(strtrim(string(x(i))));
      if any(t == ["german","deutsch","de","deu"])
          y(i) = "german";
      elseif any(t == ["english","eng","en"])
          y(i) = "english";
      else
          y(i) = t;
      end
  end
end

function L = canon_lang_from_path(fn)
  fn_norm = lower(strrep(fn,'/','\'));
  parts = string(strsplit(fn_norm, filesep));
  if any(ismember(parts, ["german","deutsch","de","deu"]))
      L = 'german';
  elseif any(ismember(parts, ["english","eng","en"]))
      L = 'english';
  else
      L = 'unknown';
  end
end

function [language, video_id] = parse_lang_vid(pathStr, baseDir)
  rel = pathStr;
  prefix = [strrep(baseDir,'/','\') filesep];
  if strncmpi(strrep(pathStr,'/','\'), prefix, numel(prefix))
      rel = pathStr(numel(prefix)+1:end);
  end
  rel = strrep(rel,'/','\');
  parts = strsplit(rel, filesep);
  if numel(parts) >= 2
      language = lower(string(parts{1}));
      video_id = string(parts{2});
  else
      language = "unknown";
      [~, video_id] = fileparts(fileparts(pathStr));
      video_id = string(video_id);
  end
end

function im = read_resize_jpeg(fname, imSize)
  data = vl_imreadjpeg({fname}, 'Resize', imSize, 'Interpolation', 'bilinear');
  im = data{1};
  if size(im,3)==1, im = repmat(im,[1 1 3]); end
end

function s = fmt_int(n)
  s = regexprep(sprintf('%,d', n), ',', '{,}');
end

function h = string2hash(str)
  s = uint32(5381);
  c = uint8(char(str));
  for k = 1:numel(c)
      s = bitand(uint32(33)*s + uint32(c(k)), uint32(2^32-1));
  end
  h = double(s);
end

function [TR, HV] = split_heardval_lang(Tlang, frac)
  if isempty(Tlang), TR = Tlang; HV = Tlang; return; end
  sc = zeros(height(Tlang),1);
  for i=1:height(Tlang)
      sc(i) = mod(string2hash(Tlang.wav_path{i}), 1e6) / 1e6; %#ok<AGROW>
  end
  HVmask = false(height(Tlang),1);
  ids_u = unique(Tlang.celebrity);
  for ui = 1:numel(ids_u)
      idx = strcmp(Tlang.celebrity, ids_u{ui});
      if ~any(idx), continue; end
      sel = sc(idx) < frac;
      if ~any(sel) && nnz(idx) >= 2
          [~,minj] = min(sc(idx));
          sel(minj) = true;
      end
      HVmask(idx) = sel;
  end
  HV = Tlang(HVmask,:);
  TR = Tlang(~HVmask,:);
end




