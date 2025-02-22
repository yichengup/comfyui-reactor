import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import torch
import time

class VideoToFrames:
    def __init__(self):
        self.type = "VideoToFrames"
        self.output_dir = "temp/frames"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING",), # 接收视频文件路径
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO",) # frames, fps, audio_path
    RETURN_NAMES = ("frames", "fps", "audio")
    FUNCTION = "process_video"
    CATEGORY = "🌌 ReActor"
    
    def process_video(self, video, select_every_nth):
        # 检查视频文件是否存在
        if not os.path.exists(video):
            print(f"警告：视频文件不存在 - {video}")
            return None

        # 读取视频
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 - {video}")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 优化的抽帧算法 - 使用顺序读取
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % select_every_nth == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转换为 PyTorch tensor 格式
                frame = torch.from_numpy(frame).float() / 255.0  # 归一化到 0-1
                frames.append(frame)  # 保持 HWC 格式
            
            frame_idx += 1
        
        cap.release()
                
        # 提取音频
        try:
            video_clip = VideoFileClip(video)
            if video_clip.audio is not None:
                audio = os.path.join(self.output_dir, "audio.mp3")
                video_clip.audio.write_audiofile(audio, verbose=False, logger=None)
            else:
                print(f"警告：视频没有音频轨道 - {video}")
                audio = None
            video_clip.close()
        except Exception as e:
            print(f"错误：音频提取失败 - {str(e)}")
            audio = None
        
        # 将帧列表转换为张量堆叠，保持 BHWC 格式
        if not frames:
            print("错误：没有成功提取到视频帧")
            return None
            
        frames = torch.stack(frames)  # 自动堆叠为 BHWC
        
        return (frames, fps, audio)

class FramesToVideo:
    def __init__(self):
        self.type = "FramesToVideo"
        self.output_dir = "temp/output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),  # 接收图像序列
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
                "audio": ("AUDIO", {"default": None}),  # 音频文件路径
                "format": (["video/h264-mp4", "video/vp9-webm"], {"default": "video/h264-mp4"}),  # 视频格式
                "pix_fmt": (["yuv420p", "yuv420p10le"], {"default": "yuv420p"}),  # 像素格式
                "crf": ("INT", {"default": 19, "min": 0, "max": 100, "step": 1}),  # 视频质量
                "save_metadata": ("BOOLEAN", {"default": True}),  # 是否保存元数据
                "trim_to_audio": ("BOOLEAN", {"default": False}),  # 是否裁剪到音频长度
            }
        }
    
    RETURN_TYPES = ("VHS_FILENAMES",)  # 返回VHS文件名格式
    RETURN_NAMES = ("filenames",)
    FUNCTION = "create_video"
    CATEGORY = "🌌 ReActor"
    
    def create_video(self, frames, fps, audio=None, format="video/h264-mp4", pix_fmt="yuv420p", 
                    crf=19, save_metadata=True, trim_to_audio=False):
        # 生成输出文件路径
        ext = ".mp4" if format == "video/h264-mp4" else ".webm"
        output_path = os.path.join(self.output_dir, f"output_{int(time.time())}{ext}")
        temp_video = os.path.join(self.output_dir, f"temp_{int(time.time())}{ext}")
        
        try:
            # 确保frames是numpy数组格式
            if torch.is_tensor(frames):
                frames = (frames * 255).byte().cpu().numpy()
            
            # 获取视频尺寸
            height, width = frames[0].shape[:2] if len(frames.shape) == 4 else frames.shape[1:3]
            
            # 创建临时帧目录
            temp_frame_dir = os.path.join(self.output_dir, f"temp_frames_{int(time.time())}")
            os.makedirs(temp_frame_dir, exist_ok=True)
            
            # 保存帧为图片
            frame_files = []
            for i, frame in enumerate(frames):
                # 转换为BGR格式
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # 确保像素值在0-255范围内
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                frame_path = os.path.join(temp_frame_dir, f"frame_{i:05d}.png")
                cv2.imwrite(frame_path, frame)
                frame_files.append(frame_path)
            
            # 使用ffmpeg进行高质量编码
            if format == "video/h264-mp4":
                codec_params = [
                    "-c:v", "libx264",
                    "-pix_fmt", pix_fmt,
                    "-crf", str(crf),
                    # 色彩空间设置
                    "-vf", "colorspace=all=bt709:iall=bt601-6-625:fast=1",
                    "-colorspace", "1",
                    "-color_primaries", "1",
                    "-color_trc", "1"
                ]
                if save_metadata:
                    codec_params.extend(["-movflags", "+faststart"])
            else:  # webm
                codec_params = [
                    "-c:v", "libvpx-vp9",
                    "-crf", "30",
                    "-b:v", "0",
                    "-pix_fmt", pix_fmt
                ]
            
            # 构建ffmpeg命令
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_frame_dir, "frame_%05d.png"),
                *codec_params,
                temp_video
            ]
            
            # 执行ffmpeg命令
            subprocess.run(cmd, check=True)
            
            # 如果有音频，合并音频
            if audio is not None and os.path.exists(audio):
                try:
                    # 创建视频和音频剪辑
                    video = VideoFileClip(temp_video)
                    audio_clip = AudioFileClip(audio)
                    
                    # 如果需要裁剪到音频长度
                    if trim_to_audio:
                        video = video.set_duration(audio_clip.duration)
                    
                    # 合并视频和音频
                    final_video = video.set_audio(audio_clip)
                    final_video.write_videofile(output_path, 
                                             codec='libx264' if format == "video/h264-mp4" else 'libvpx-vp9',
                                             audio_codec='aac',
                                             verbose=False,
                                             logger=None)
                    
                    # 清理
                    video.close()
                    audio_clip.close()
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
                except Exception as e:
                    print(f"警告：音频合并失败 - {str(e)}")
                    # 如果音频合并失败，使用无音频的视频
                    os.rename(temp_video, output_path)
            else:
                # 如果没有音频，直接使用生成的视频
                os.rename(temp_video, output_path)
            
            # 清理临时文件
            import shutil
            if os.path.exists(temp_frame_dir):
                shutil.rmtree(temp_frame_dir)
            
            # 返回成功状态和文件路径列表
            return ((True, [output_path]),)  # 返回VHS_FILENAMES格式
            
        except Exception as e:
            print(f"错误：视频生成失败 - {str(e)}")
            # 清理临时文件
            if os.path.exists(temp_frame_dir):
                shutil.rmtree(temp_frame_dir)
            # 返回失败状态和空列表
            return ((False, []),)