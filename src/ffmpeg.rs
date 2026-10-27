//! ffmpeg encoding logic
use crate::{
    command::args::PixelFormat,
    float::TerseF32,
    process::{Chunks, CommandExt, FfmpegOut, FfmpegOutStream, exit_ok_stderr},
    temporary::{self, TempKind},
};
use anyhow::Context;
use bstr::ByteSlice;
use log::{debug, info};
use std::{
    collections::HashSet,
    fmt::Write,
    hash::{Hash, Hasher},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, LazyLock},
};
use tokio::process::Command;
use tokio_process_stream::{Item, ProcessChunkStream};
use tokio_stream::StreamExt;

/// Exposed ffmpeg encoding args.
#[derive(Debug, Clone)]
pub struct FfmpegEncodeArgs<'a> {
    pub input: &'a Path,
    pub vcodec: Arc<str>,
    pub vfilter: Option<&'a str>,
    pub pix_fmt: Option<PixelFormat>,
    pub crf: f32,
    pub preset: Option<Arc<str>>,
    pub output_args: Vec<Arc<String>>,
    pub input_args: Vec<Arc<String>>,
    pub video_only: bool,
}

impl FfmpegEncodeArgs<'_> {
    pub fn sample_encode_hash(&self, state: &mut impl Hasher) {
        static SVT_AV1_V: LazyLock<String> = LazyLock::new(|| {
            ffmpeg_svtav1_version()
                .inspect_err(|e| debug!("read_ffmpeg_svtav1_version: {e}"))
                .unwrap_or_default()
        });

        // hashing svt-av1 version means new encoder releases will avoid old cache data
        if &*self.vcodec == "libsvtav1" {
            SVT_AV1_V.hash(state);
        }

        // input not relevant to sample encoding
        self.vcodec.hash(state);
        self.vfilter.hash(state);
        self.pix_fmt.hash(state);
        self.crf.to_bits().hash(state);
        self.preset.hash(state);
        self.output_args.hash(state);
        self.input_args.hash(state);
    }
}

/// Encode a sample.
pub fn encode_sample(
    FfmpegEncodeArgs {
        input,
        vcodec,
        vfilter,
        pix_fmt,
        crf,
        preset,
        output_args,
        input_args,
        video_only: _,
    }: FfmpegEncodeArgs,
    temp_dir: Option<PathBuf>,
    dest_ext: &str,
) -> anyhow::Result<(PathBuf, FfmpegOutStream)> {
    let pre = pre_extension_name(&vcodec);
    let crf_str = format!("{}", TerseF32(crf)).replace('.', "_");
    let dest_file_name = match &preset {
        Some(p) => input.with_extension(format!("{pre}.crf{crf_str}.{p}.{dest_ext}")),
        None => input.with_extension(format!("{pre}.crf{crf_str}.{dest_ext}")),
    };
    let dest_file_name = dest_file_name.file_name().unwrap();
    let mut dest = temporary::process_dir(temp_dir)?;
    dest.push(dest_file_name);

    temporary::add(&dest, TempKind::Keepable);

    let mut cmd = Command::new("ffmpeg");
    cmd.kill_on_drop(true)
        .arg("-y")
        .args(input_args.iter().map(|a| &**a))
        .arg2("-i", input)
        .arg2("-c:v", &*vcodec)
        .args(output_args.iter().map(|a| &**a))
        // Avoid dropping or duplicating frames as this may negatively affect input/output analysis
        .arg2("-fps_mode", "passthrough")
        .arg2(vcodec.crf_arg(), vcodec.crf(crf))
        .arg2_opt("-pix_fmt", pix_fmt.map(|v| v.as_str()))
        .arg2_opt(vcodec.preset_arg(), preset)
        .arg2_opt("-vf", vfilter)
        .arg("-an")
        .arg(&dest)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let cmd_str = cmd.to_cmd_str();
    debug!("cmd `{cmd_str}`");

    let enc = cmd.spawn().context("ffmpeg encode_sample")?;

    let stream = FfmpegOut::stream(enc, "ffmpeg encode_sample", cmd_str);
    Ok((dest, stream))
}

/// Encode to output.
pub fn encode(
    FfmpegEncodeArgs {
        input,
        vcodec,
        vfilter,
        pix_fmt,
        crf,
        preset,
        output_args,
        input_args,
        video_only,
    }: FfmpegEncodeArgs,
    output: &Path,
    has_audio: bool,
    audio_codec: Option<&str>,
    downmix_to_stereo: bool,
    fail_fast: bool,
) -> anyhow::Result<FfmpegOutStream> {
    let oargs: HashSet<_> = output_args.iter().map(|a| a.as_str()).collect();
    let output_ext = output.extension().and_then(|e| e.to_str());

    let add_faststart = output_ext == Some("mp4") && !oargs.contains("-movflags");
    let matroska = matches!(output_ext, Some("mkv") | Some("webm"));
    let add_cues_to_front = matroska && !oargs.contains("-cues_to_front");

    let audio_codec = audio_codec.unwrap_or(if downmix_to_stereo && has_audio {
        "libopus"
    } else {
        "copy"
    });

    let set_ba_128k = audio_codec == "libopus" && !oargs.contains("-b:a");
    let downmix_to_stereo = downmix_to_stereo && !oargs.contains("-ac");
    let map = match video_only {
        true => "0:v:0",
        false => "0",
    };
    // This doesn't seem to work on .mp4 files
    let mut metadata = format!(
        "AB_AV1_FFMPEG_ARGS=-c:v {vcodec} {} {crf}",
        vcodec.crf_arg()
    );
    if let Some(preset) = &preset {
        write!(&mut metadata, " {} {preset}", vcodec.preset_arg()).unwrap();
    }

    let mut cmd = Command::new("ffmpeg");
    cmd.kill_on_drop(true)
        .args(input_args.iter().map(|a| &**a))
        .arg("-y")
        .arg_if(fail_fast, "-xerror")
        .arg2("-i", input)
        .arg2("-map", map)
        .arg2("-c:v", "copy")
        .arg2("-c:v:0", &*vcodec)
        .arg2("-metadata", metadata)
        .arg2("-c:a", audio_codec)
        .arg2("-c:s", "copy")
        .args(output_args.iter().map(|a| &**a))
        .arg2(vcodec.crf_arg(), vcodec.crf(crf))
        .arg2_opt("-pix_fmt", pix_fmt.map(|v| v.as_str()))
        .arg2_opt(vcodec.preset_arg(), preset)
        .arg2_opt("-vf", vfilter)
        .arg_if(matroska, "-dn") // "Only audio, video, and subtitles are supported for Matroska"
        .arg2_if(downmix_to_stereo, "-ac", 2)
        .arg2_if(set_ba_128k, "-b:a", "128k")
        .arg2_if(add_faststart, "-movflags", "+faststart")
        .arg2_if(add_cues_to_front, "-cues_to_front", "y")
        .arg(output)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let cmd_str = cmd.to_cmd_str();
    debug!("cmd `{cmd_str}`");

    let enc = cmd.spawn().context("ffmpeg encode")?;

    Ok(FfmpegOut::stream(enc, "ffmpeg encode", cmd_str))
}

pub fn pre_extension_name(vcodec: &str) -> &str {
    match vcodec.strip_prefix("lib").filter(|s| !s.is_empty()) {
        Some("svtav1") => "av1",
        Some("vpx-vp9") => "vp9",
        Some(suffix) => suffix,
        _ => vcodec,
    }
}

trait VCodecSpecific {
    /// Arg to use preset values with, normally `-preset`.
    fn preset_arg(&self) -> &str;
    /// Arg to use crf values with, normally `-crf`.
    fn crf_arg(&self) -> &str;
    /// crf value to pass to ffmpeg.
    fn crf(&self, crf: f32) -> f32;
}
impl VCodecSpecific for Arc<str> {
    fn preset_arg(&self) -> &str {
        match &**self {
            "libaom-av1" | "libvpx-vp9" => "-cpu-used",
            "librav1e" => "-speed",
            _ => "-preset",
        }
    }

    fn crf_arg(&self) -> &str {
        // use crf-like args to support encoders that don't have crf
        match &**self {
            // https://ffmpeg.org//ffmpeg-codecs.html#librav1e
            // https://github.com/fraunhoferhhi/vvenc/wiki/FFmpeg-Integration#fix-qp-mode-constant-quality-mode
            "librav1e" | "libvvenc" => "-qp",
            "mpeg2video" => "-q",
            "hevc_videotoolbox" => "-q:v",
            // https://ffmpeg.org//ffmpeg-codecs.html#VAAPI-encoders
            e if e.ends_with("_vaapi") => "-q",
            e if e.ends_with("_vulkan") => "-qp",
            e if e.ends_with("_nvenc") => "-cq",
            // https://ffmpeg.org//ffmpeg-codecs.html#QSV-Encoders
            e if e.ends_with("_qsv") => "-global_quality",
            _ => "-crf",
        }
    }

    fn crf(&self, crf: f32) -> f32 {
        match &**self {
            // ffmpeg svt-av1 crf above 63 don't work, but up to 70 does work in -svtav1-params
            "libsvtav1" => crf.min(63.0),
            _ => crf,
        }
    }
}

pub fn remove_arg(args: &mut Vec<Arc<String>>, arg: &'static str) {
    let mut retain_next = true;
    args.retain(|a| {
        if **a == arg {
            retain_next = false;
            false
        } else if !retain_next {
            retain_next = true;
            false
        } else {
            true
        }
    });
}

fn ffmpeg_svtav1_version() -> anyhow::Result<String> {
    let mut ffmpeg = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "info",
            "-f",
            "lavfi",
            "-i",
            "color",
            "-frames:v",
            "1",
            "-c:v",
            "libsvtav1",
            "-f",
            "null",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stderr = BufReader::new(ffmpeg.stderr.take().context("stderr")?);

    std::thread::spawn(move || _ = ffmpeg.wait());

    read_svtav1_version(&mut stderr)
}

/// Read ffmpeg stderr until the svt-av1 version line is found.
fn read_svtav1_version(reader: &mut impl BufRead) -> anyhow::Result<String> {
    // Example: "SVT-AV1 Encoder Lib v123.456.678"
    const PREFIX: &str = "SVT-AV1 Encoder Lib ";

    let mut line = Vec::new();
    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line)?;
        anyhow::ensure!(n != 0, "EOF: No version string found");

        let Some(idx) = line.find_iter(PREFIX).next() else {
            continue;
        };

        let ver: Vec<_> = line[idx + PREFIX.len()..]
            .iter()
            .copied()
            .take_while(|b| !b.is_ascii_whitespace())
            .collect();

        if !ver.is_empty() {
            let version = String::try_from(ver)?;
            println!("found svt-av1 version: {version}");
            return Ok(version);
        }
    }
}

/// Decode all video & audio streams of `file`, failing on the first decode error.
///
/// ffmpeg can exit successfully having written a damaged or short result, usually
/// because the input itself is damaged. Neither the exit code nor sample based VMAF
/// notices this, since the samples may all land on healthy parts of the file.
///
/// `on_progress` is called with the decode progress, so callers can keep a progress
/// bar moving during what is a full pass over the file.
pub async fn decode(file: &Path, mut on_progress: impl FnMut(FfmpegOut)) -> anyhow::Result<()> {
    info!(
        "verifying {}",
        file.file_name().and_then(|n| n.to_str()).unwrap_or("")
    );

    let mut cmd = Command::new("ffmpeg");
    cmd.kill_on_drop(true)
        // Only errors are of interest, `-stats` keeps the progress output that the
        // "error" log level would otherwise hide.
        .arg2("-v", "error")
        .arg("-stats")
        .arg("-xerror")
        .arg2("-i", file)
        // Decode all video & audio streams, not just the ones ffmpeg would select by
        // default. Subtitles, attachments & data are skipped as the null muxer has no
        // encoder for them, and the encode copies them through unchanged anyway.
        .arg2("-map", "0:v?")
        .arg2("-map", "0:a?")
        .arg2("-f", "null")
        .arg("-")
        .stdin(Stdio::null());

    let cmd_str = cmd.to_cmd_str();
    debug!("cmd `{cmd_str}`");
    let mut dec = crate::process::child::AddOnDropChunkStream::from(
        ProcessChunkStream::try_from(cmd).context("ffmpeg verify")?,
    );

    let mut chunks = Chunks::default();
    while let Some(next) = dec.next().await {
        match next {
            Item::Stderr(chunk) => {
                chunks.push(&chunk);
                if let Some(out) = FfmpegOut::try_parse(chunks.last_line()) {
                    on_progress(out);
                }
            }
            Item::Stdout(_) => {}
            Item::Done(code) => exit_ok_stderr("ffmpeg verify", code, &cmd_str, &chunks)?,
        }
    }

    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{
        env, fs,
        os::unix::fs::PermissionsExt,
        path::{Path, PathBuf},
        sync::{Mutex, MutexGuard},
    };

    static PATH_LOCK: Mutex<()> = Mutex::new(());

    struct PathGuard(Option<std::ffi::OsString>);

    impl PathGuard {
        fn prepend(dir: &Path) -> Self {
            let original = env::var_os("PATH");
            let mut paths = vec![dir.to_path_buf()];
            if let Some(path) = &original {
                paths.extend(env::split_paths(path));
            }
            let path = env::join_paths(paths).expect("join PATH");
            // SAFETY: fake-ffmpeg tests are serialized by PATH_LOCK.
            unsafe { env::set_var("PATH", path) };
            Self(original)
        }
    }

    impl Drop for PathGuard {
        fn drop(&mut self) {
            // SAFETY: fake-ffmpeg tests are serialized by PATH_LOCK.
            unsafe {
                match &self.0 {
                    Some(path) => env::set_var("PATH", path),
                    None => env::remove_var("PATH"),
                }
            }
        }
    }

    struct FakeFfmpeg {
        dir: PathBuf,
        _path: PathGuard,
        _lock: MutexGuard<'static, ()>,
    }

    impl FakeFfmpeg {
        fn new(script: &str) -> Self {
            let lock = PATH_LOCK.lock().unwrap();
            let dir = env::temp_dir().join(format!(
                "ab-av1-fake-ffmpeg-{}-{}",
                std::process::id(),
                fastrand::u64(..)
            ));
            fs::create_dir_all(&dir).expect("create fake ffmpeg dir");

            let ffmpeg = dir.join("ffmpeg");
            fs::write(&ffmpeg, script).expect("write fake ffmpeg");
            let mut perms = fs::metadata(&ffmpeg).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&ffmpeg, perms).unwrap();

            let path = PathGuard::prepend(&dir);
            Self {
                dir,
                _path: path,
                _lock: lock,
            }
        }
    }

    impl Drop for FakeFfmpeg {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn fake_ffmpeg_version(script: &str) -> anyhow::Result<String> {
        let _fake = FakeFfmpeg::new(script);
        ffmpeg_svtav1_version()
    }

    /// The old parser only accepted a version once a later byte was seen, so a
    /// stream ending on a version digit hit EOF and reported no version.
    #[test]
    fn ffmpeg_svtav1_version_at_eof() {
        let version =
            fake_ffmpeg_version("#!/bin/sh\nprintf 'SVT-AV1 Encoder Lib v1.2.3' >&2\n").unwrap();

        assert_eq!(version, "v1.2.3");
    }

    /// Reproduces the intermittent real-world failure: the old parser rotated
    /// its 256-byte buffer after a read ending in a version character, losing
    /// the version line before it could parse it.
    #[test]
    fn ffmpeg_svtav1_version_across_stderr_chunks() {
        let script = r#"#!/bin/sh
printf '%100s' '' | tr ' ' 'x' >&2
sleep 0.1
printf 'SVT-AV1 Encoder Lib v4.2.0-210-gf2c3fb786' >&2
sleep 0.1
printf '%99s9' '' >&2
sleep 0.1
printf '\n' >&2
"#;

        let version = fake_ffmpeg_version(script).unwrap();

        assert_eq!(version, "v4.2.0-210-gf2c3fb786");
    }
}
