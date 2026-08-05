#!/usr/bin/env python
"""Build golden_extended.json = golden_100 + 8 image questions + 15 Ecovacs questions.

Image questions use the VLM descriptions of the 6 Roborock diagram pages
(p5/6/7/8/10/23) as reference_contexts. Ecovacs questions use Chinese questions
with verbatim English contexts (tests cross-lingual retrieval via BGE-M3).

Usage: python scripts/build_extended_dataset.py
Output: data/eval_dataset/golden_extended.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _img(page: int, summary: str, q: str, ans: str, qtype: str = "feature",
         diff: str = "medium") -> dict:
    ctx = f"[图片语义描述 - 第{page}页]\n{summary}"
    return {
        "question": q,
        "question_type": qtype,
        "difficulty": diff,
        "modality_required": "image",
        "gold_pages": [page],
        "reference_answer": ans,
        "reference_context": ctx,
        "reference_contexts": [ctx],
        "source_document": "Roborock G10S",
        "review_status": "ai_annotated",
    }


# ── 8 image questions (Roborock diagram pages) ──

P5 = "机器人正面包含：清扫/开关机键（短按清扫、长按开关机）、回充键（短按回充、长按回洗拖布）、局部清扫/童锁键（短按局部清扫、长按3秒童锁）。电源指示灯：白色电量≥20%、红色<20%、呼吸闪烁充电中、红色快闪异常。"
P6 = "机器人底部装有悬崖传感器，可检测楼梯、台阶等高差环境，防止机器人跌落。地毯识别传感器自动识别地毯并触发拖布支架升降。底部还有万向轮、主轮、边刷、软胶主刷、集尘进风口。碰撞缓冲传感器位于前端。"
P7 = "尘盒包含上置可水洗滤网和滤网上盖。电控水箱通过进水口滤网和自动补水口与基座连接。震动拖布支架不可拆卸，拖布通过插槽和粘贴区域固定。升降震动擦地模组提供擦地动力。"
P8 = "基座包含：污水箱（上盖+卡扣）、清水箱（提手+上盖）、集尘桶（内置一次性尘袋）、充电弹片、高速自清洁刷（可拆卸卡扣）、清洗槽滤网。状态指示灯：呼吸闪烁=集尘/清洁拖布中、红色=异常、熄灭=未通电或充电中。"
P10 = "基座安装步骤：1)取出底部高速自清洁刷组件运输固定泡沫；2)基座底板与主体组装，下压听到咔哒声；3)电源线插紧，多余线材收入线槽。基座放置：硬质水平地面靠墙，两侧0.5米、前方1.5米、上方1米以上空间。"
P23 = "报废前拆电池步骤：1)不接触基座运行至低电量；2)关机；3)卸下电池盖板螺丝；4)取下盖板；5)按下卡扣拔出连接器插头取下电池。注意：确保电量用尽、断开基座、整组拆卸、勿损坏外壳。渗出物接触皮肤用大量清水冲洗并及时就医。"

IMAGE_QUESTIONS = [
    _img(5, P5, "机器人的回充键长按有什么功能？", "长按回洗拖布。", qtype="feature"),
    _img(5, P5, "机器人电源指示灯红色快闪代表什么？", "异常状态。", qtype="troubleshooting", diff="easy"),
    _img(6, P6, "机器人的碰撞缓冲传感器安装在哪个部位？", "前端。", qtype="feature", diff="easy"),
    _img(6, P6, "机器人底部安装了哪些部件？", "悬崖传感器、地毯识别传感器、万向轮、主轮、边刷、软胶主刷、集尘进风口。", qtype="feature"),
    _img(7, P7, "尘盒的滤网安装在上方还是下方？可以水洗吗？", "上置，可以水洗。", qtype="maintenance", diff="easy"),
    _img(8, P8, "基座的高速自清洁刷可以拆卸吗？", "可以，通过卡扣固定，可拆卸。", qtype="maintenance", diff="easy"),
    _img(10, P10, "基座电源线插紧后，多余线材应该怎么处理？", "收入基座线槽。", qtype="setup", diff="easy"),
    _img(23, P23, "取下电池时需要注意哪些事项？", "确保电量用尽、断开基座、整组拆卸、勿损坏电池组外壳。", qtype="maintenance"),
]


# ── 15 Ecovacs questions (Chinese Q + English reference contexts) ──

def _eco(pages: list[int], q: str, ans: str, ctx: list[str],
          qtype: str, diff: str = "medium") -> dict:
    return {
        "question": q,
        "question_type": qtype,
        "difficulty": diff,
        "modality_required": "text",
        "gold_pages": pages,
        "reference_answer": ans,
        "reference_context": ctx[0],
        "reference_contexts": ctx,
        "source_document": "Ecovacs DEEBOT T30C",
        "review_status": "ai_annotated",
    }


ECOVACS_QUESTIONS = [
    _eco([3], "这款扫地机器人允许儿童当作玩具使用吗？",
         "不可以，使用时需在儿童身边密切看护。",
         ["Do not allow to be used as a toy. Close attention is necessary when used by or near children."],
         "safety", "easy"),
    _eco([15], "机器人可以在湿滑地面或有积水的表面上使用吗？",
         "不可以，禁止在湿表面或积水地面使用。",
         ["Do not use the appliance on wet surfaces or surfaces with standing water."],
         "safety", "easy"),
    _eco([15], "部件安装到位的标志是什么声音？",
         "听到\"咔哒\"声表示安装到位。",
         ["The sound of \"click\" indicates proper installation."],
         "setup", "easy"),
    _eco([16, 26], "机器人连接 App 需要什么网络条件？",
         "需要家庭 Wi-Fi 网络，不支持 5GHz 频段，且 Wi-Fi 名称不能含特殊字符。",
         ["The robot series robotic vacuums are designed for domestic cleaning scenarios. A home Wi-Fi network is required for operation.",
          "Do not use a 5 Ghz network.",
          "Check if the Wi-Fi name contains special characters. Please do not use special characters like ! @#& ¥%\\."],
         "setup"),
    _eco([9], "包装箱内含哪些组件？",
         "机器人、OMNI 基座、说明书、边刷、基座底板、电源线。",
         ["Robot, OMNI Station, Instruction Manual, Side Brush, Base, Power Cord"],
         "feature", "easy"),
    _eco([12], "机器人指示灯白色呼吸闪烁代表什么状态？",
         "正在充电。",
         ["Breathing White — Charging", "Solid White — Fully Charged/Running", "Solid Red — Low Battery", "Flashing Red — Alarm"],
         "feature", "easy"),
    _eco([13], "导航模块的激光测距检测范围是多少？",
         "8 米。",
         ["Navigation Module: Laser Ranging is applied to measure the distance between the robot and surrounding objects... The detection range is 8 m."],
         "feature", "easy"),
    _eco([20], "主刷建议多久清理一次？多久更换？",
         "每周清理，每 6-12 个月更换。",
         ["Main Brush — Every week — Every 6-12 months"],
         "maintenance", "easy"),
    _eco([20], "边刷建议多久更换？",
         "每 3-6 个月。",
         ["Side Brush — Every 2 weeks — Every 3-6 months"],
         "maintenance", "easy"),
    _eco([21], "滤网可以用手指或刷子清理吗？",
         "不可以，应水洗滤网并完全晾干后使用，禁止用手指或刷子。",
         ["Do not use fingers or a brush to clean the filter.",
          "Please rinse the filter with water as shown.",
          "Completely dry the filter before use."],
         "maintenance", "easy"),
    _eco([22], "如何清理主刷？",
         "打开上盖，取下主刷清理；取下梳刷清理并擦干；装回梳刷、主刷和主刷盖。",
         ["Clean the Roller Brush: 1. Open the cover. 2. Remove and clean the main brush. 3. Remove the brush comb, clean it, then wipe it dry. 4. Install the brush comb, the main brush and the main brush cover."],
         "maintenance"),
    _eco([26], "机器人无法连接 App 的常见原因有哪些？",
         "Wi-Fi 名称或密码错误、机器人不在 Wi-Fi 信号范围内、未处于配网状态、Wi-Fi 名称含特殊字符或使用 5GHz 网络。",
         ["Incorrect Wi-Fi name or password entered. Enter the correct Wi-Fi username and password.",
          "The robot is not within Wi-Fi signal coverage.",
          "The robot is not in the configuration state.",
          "Do not use a 5 Ghz network.",
          "Check if the Wi-Fi name contains special characters."],
         "troubleshooting"),
    _eco([28], "机器人回站后不自动集尘的原因有哪些？",
         "集尘舱未关闭、App 未开启自动集尘功能、尘袋未安装、手动搬回基座、或处于勿扰模式。",
         ["Dust collection cabin is not closed.",
          "The Auto-Empty function has not been turned on in the ECOVACS HOME App.",
          "Dust bag is not installed in the station.",
          "Manually moving the robot back to the station might not trigger the auto-empty feature.",
          "In Do Not Disturb mode, the robot will not empty the dust bin after returning to the station."],
         "troubleshooting"),
    _eco([29], "驱动轮卡住如何处理？",
         "旋转并按压驱动轮，检查并清除缠绕卡住的异物；若问题持续，联系客服。",
         ["The driving wheels are entangled or jammed by foreign objects. Please rotate and press the driving wheels to check for and remove any foreign objects that are entangled or jammed. If this problem persists, please contact customer care for help."],
         "troubleshooting"),
    _eco([30], "机器人的充电时间大约多久？",
         "约 4.5 小时。",
         ["Charging Time — about 4.5 h"],
         "feature", "easy"),
]


def main() -> None:
    base = json.loads(
        (PROJECT_ROOT / "data" / "eval_dataset" / "golden_100.json")
        .read_text(encoding="utf-8")
    )
    all_q = base + IMAGE_QUESTIONS + ECOVACS_QUESTIONS

    out = PROJECT_ROOT / "data" / "eval_dataset" / "golden_extended.json"
    out.write_text(json.dumps(all_q, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    print(f"Total: {len(all_q)} questions")
    print("  source_document:", dict(Counter(q["source_document"] for q in all_q)))
    print("  modality:", dict(Counter(q.get("modality_required", "text") for q in all_q)))
    print("  question_type:", dict(Counter(q["question_type"] for q in all_q)))
    print("Saved:", out)


if __name__ == "__main__":
    main()
