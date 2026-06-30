from __future__ import annotations

import json
from typing import Any

from .schemas import Clip

GOOSE_COMMON_KNOWLEDGE = """共同知识：
- 《鹅鸭杀》是社交推理游戏。鹅阵营通常通过做任务、分享观察、投票放逐鸭来获胜；鸭阵营通过击杀、破坏、误导和制造错误怀疑来获胜；中立角色可能有独立胜利条件。
- 会议、报警、尸体报告、投票结果、公开发言通常是公共信息；单个 POV 看到的路径、任务界面、附近玩家、局部语音和未公开发现是私有信息。
- 地图房间名、任务面板、传送/管道/门、破坏任务、尸体位置、相遇顺序、谁能看见谁，都是后续心智状态推理的重要依据。
- 任务 UI、点击连线、小游戏界面、鼠标操作只说明玩家正在交互或做任务，不等同于角色在地图上移动。只有画面中角色位置变化或房间切换才标 movement。
- 标注时区分可见事实、语音声明、推测/怀疑和不确定信息。不要把玩家口头指认当作已证实事实。
- 对地点、任务和机制不确定时，用 uncertainty 或 evidence 说明不确定，不要硬编地图细节。
"""


def _clip_context(clip: Clip) -> str:
    return (
        f"game_id={clip.game_id}, player_id={clip.player_id}, "
        f"clip_id={clip.clip_id}, aligned_time=[{clip.start_sec}, {clip.end_sec}]"
    )


def pov_event_prompt(clip: Clip) -> str:
    return f"""你是《鹅鸭杀》多视角录像的结构化标注员。
请细致分析这个 POV 短片段，抽取玩家可见或可听到的关键事件，用于后续 Theory-of-Mind benchmark。

{GOOSE_COMMON_KNOWLEDGE}

片段信息：{_clip_context(clip)}

请重点观察：
1. 玩家当前位置、真实移动路线、任务完成、地图/房间切换。
2. 其他玩家是否出现在视野中，出现/离开的大致时间和位置。
3. 击杀、尸体、报警、投票、会议、怀疑、辩解、指认、结盟或欺骗线索。
4. 语音中说了什么，谁在说，提到了谁，哪些信息是公开的，哪些只对当前 POV 可见。
5. 当前玩家可能知道/相信/不知道什么；不要把片段外的信息写成事实。
6. 如果画面是任务小游戏或点击连线，请标为 task_ui 或 interaction，并说明正在操作 UI，不要写成“前往某处”。

只输出 JSON 数组，不要输出 Markdown。每个元素必须包含：
clip_id, game_id, player_id, start_sec, end_sec, event_type, description,
visible_players, mentioned_players, location, confidence, evidence。
时间必须使用对齐后的全局秒数，且落在片段时间范围内。
最多输出 8 个关键事件；visible_players 和 mentioned_players 各最多 5 个，去重后输出。
visible_players 只写当前画面真实同屏或 UI 明确显示且与事件相关的玩家，不要列会议全员名单。
mentioned_players 只写语音或文字中明确提到且与事件相关的玩家；不确定姓名时宁可写空列表。
不要把同一个玩家名重复填入列表；不要补全、猜测或生成长名单。
description 和 evidence 每项控制在 1-2 句，避免冗长。
event_type 使用简洁英文，例如 movement, task, encounter, meeting, accusation,
defense, vote, kill, body_report, role_clue, uncertainty, audio_cue, task_ui, interaction。
description 和 evidence 用中文，尽量具体。
如果没有可靠事件，输出 []。
"""


def utterance_prompt(clip: Clip) -> str:
    return f"""你是《鹅鸭杀》会议与语音转写标注员。
请识别这个 POV 短片段中的发言，重点关注会议讨论、指认、辩解和投票相关话语。

{GOOSE_COMMON_KNOWLEDGE}

片段信息：{_clip_context(clip)}

只输出 JSON 数组，不要输出 Markdown。每个元素必须包含：
game_id, player_id, clip_id, start_sec, end_sec, speaker_id, text,
addressee_ids, mentioned_players, speech_act, confidence。
无法确定 speaker_id 时使用 null。
"""


def information_state_prompt(
    player_id: str,
    cutoff_time: float,
    prior_annotations: list[dict[str, Any]],
) -> str:
    payload = json.dumps(prior_annotations, ensure_ascii=False)
    return f"""你是 Theory-of-Mind benchmark 的信息状态构建器。
请根据截止时间前的标注，推断玩家 {player_id} 在 {cutoff_time} 秒时知道什么、相信什么、不确定什么。

{GOOSE_COMMON_KNOWLEDGE}

prior_annotations:
{payload}

只输出 JSON 对象，字段必须包含：
game_id, player_id, cutoff_time, known_facts, beliefs, uncertainties,
private_observations, public_information, confidence。
不要加入截止时间之后的信息。
"""


def global_merge_prompt(
    time_window: tuple[float, float],
    pov_annotations: list[dict[str, Any]],
) -> str:
    payload = json.dumps(pov_annotations, ensure_ascii=False)
    return f"""你是多 POV 事件融合器。
请合并同一时间窗内不同玩家视角的事件，去重并生成全局事件。

{GOOSE_COMMON_KNOWLEDGE}

time_window={time_window}
pov_annotations:
{payload}

只输出 JSON 数组。每个元素必须包含：
game_id, start_sec, end_sec, event_type, description, involved_players,
source_clip_ids, source_player_ids, confidence。
"""


def candidate_trial_prompt(
    global_events: list[dict[str, Any]],
    information_states: list[dict[str, Any]],
) -> str:
    events_payload = json.dumps(global_events, ensure_ascii=False)
    states_payload = json.dumps(information_states, ensure_ascii=False)
    return f"""你是 Theory-of-Mind benchmark 题目生成器。
请基于全局事件和玩家信息状态，生成候选测试题。

{GOOSE_COMMON_KNOWLEDGE}

global_events:
{events_payload}

information_states:
{states_payload}

只输出 JSON 数组。每个元素必须包含：
game_id, trial_id, question_type, target_player_id, cutoff_time, question,
answer, distractors, supporting_global_event_ids,
supporting_information_state_ids, difficulty, confidence。
题目必须能由给定信息支持，不能引入外部事实。
"""


def sync_start_prompt(raw_video: object) -> str:
    game_id = getattr(raw_video, "game_id")
    player_id = getattr(raw_video, "player_id")
    duration_sec = getattr(raw_video, "duration_sec", None)
    return f"""你是《鹅鸭杀》多视角录像同步员。
请观看这个原始 POV 开头审查片段，找出“第一局正式游戏开始”的原始录屏秒数 raw_start_sec。

{GOOSE_COMMON_KNOWLEDGE}

原始视频信息：
game_id={game_id}
player_id={player_id}
raw_duration_sec={duration_sec}

同步基准定义：
- raw_start_sec 应该指向第一局正式开始后所有玩家可对齐的 0 秒。
- 优先选择进入第一局地图、倒计时结束、玩家角色可操控、正式回合 UI 出现的时刻。
- 开头聊天、等待房、设置、加载、无关寒暄、直播准备、未进入正式局的部分都不算。
- 如果视频开头已经在正式局内，raw_start_sec 可以是 0，但必须说明证据。
- 如果只能粗略判断，请给低一些 confidence，并在 evidence 说明依据和不确定点。

只输出一个 JSON 对象，不要输出 Markdown。字段必须包含：
game_id, player_id, raw_start_sec, evidence, confidence。
raw_start_sec 使用原始录屏时间秒数，confidence 是 0 到 1 的数字。
"""
