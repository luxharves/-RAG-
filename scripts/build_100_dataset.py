#!/usr/bin/env python
"""Phase 2 — Build 100-question Golden Dataset from existing 20 + 80 new."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── New 80 questions (Q21-Q100) covering all PDF pages ──

NEW_QUESTIONS: list[dict] = [
    # ═══ Safety / Usage Limits (p3) ═══
    {"question":"机器人的使用温度范围是多少？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[3],"reference_answer":"机器人应在 0℃ 至 40℃ 的环境下使用。","reference_contexts":["请勿在高于 40℃、低于 0℃ 或地面有任何液体及粘稠物的环境下存放及使用。"]},
    {"question":"机器人在什么地面环境下不能使用？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[3],"reference_answer":"不能用于室外、非地面（如沙发）、商用或工业环境，不能清扫长毛地毯。","reference_contexts":["仅用于家居环境的地面清洁，请勿用于室外（如开放式阳台）、非地面（如沙发）、商用或工业环境。","请勿用于清扫长毛地毯（部分深色地毯可能无法正常清扫）。"]},
    {"question":"机器人工作时有哪些安全注意事项？","question_type":"safety","difficulty":"medium","modality_required":"text","gold_pages":[3],"reference_answer":"收起地面线缆，勿让儿童操作，勿清扫燃烧物体，勿在悬空环境使用，勿在楼梯口放低矮物品。","reference_contexts":["使用前请先将家中地面线缆收起","请勿让儿童及身体、精神或感知能力有障碍的人使用或操作本产品","请勿用于清扫任何燃烧中的物体","请勿在悬空环境（如复式楼层，开放式阳台，家具顶端）没有防护栏的前提下使用","请勿在楼梯口等可能造成机器人跌落的位置摆放地垫、鞋子等低矮物体"]},
    {"question":"可以在水箱中添加洗涤剂吗？","question_type":"troubleshooting","difficulty":"easy","modality_required":"text","gold_pages":[3],"reference_answer":"不可以，请勿在水箱中添加任何非官方清洁液或消毒剂，否则可能会造成严重损坏。","reference_contexts":["请勿在水箱中添加任何非官方清洁液/ 消毒剂，否则可能会造成机器人及基座严重损坏。"]},
    {"question":"长时间不使用机器人应如何存放？","question_type":"maintenance","difficulty":"medium","modality_required":"text","gold_pages":[3],"reference_answer":"充满电后关闭机器人，放置于阴凉干燥处，至少每3个月充电一次。","reference_contexts":["如长时间不使用机器人，请充满电后关闭机器人并放置于阴凉干燥处，至少每 3 个月充电一次避免电池出现过放。"]},
    # ═══ Product Intro / Components (p4-p8) ═══
    {"question":"机器人包装箱内包含哪些配件？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[4],"reference_answer":"包含电源线、基座主体（内置一次性尘袋）、基座底板、机器人主机、一次性尘袋。","reference_contexts":["配件清单：电源线、基座主体（内置一次性尘袋）、基座底板、机器人主机、一次性尘袋。"]},
    {"question":"电源指示灯的颜色分别代表什么含义？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[5,8,11],"reference_answer":"白色电量≥20%，红色电量<20%，呼吸闪烁表示充电或启动中，红色快闪表示异常。基座状态指示灯：呼吸闪烁=集尘/清洁拖布中，红色=异常，熄灭=未通电或充电中。","reference_contexts":["电源指示灯：白色电量≥20%、红色电量＜20%、呼吸闪烁充电或启动中、红色快闪异常状态","状态指示灯：呼吸闪烁集尘/清洁拖布中、红色基座状态异常、熄灭未通电或机器人充电中","基座指示灯通电时长亮，机器人充电时熄灭","基座出现异常后，指示灯将红灯长亮提醒"]},
    {"question":"童锁功能如何开启和关闭？","question_type":"setup","difficulty":"medium","modality_required":"text","gold_pages":[5,13],"reference_answer":"主机长按局部清扫键3秒或在手机APP中设置。童锁开启时所有按键被锁定。","reference_contexts":["长按3S 开启/关闭童锁功能","开启/关闭方式：主机长按键或在手机APP中设置。童锁开启时，机器人静止状态下所有按键均被锁定。"]},
    {"question":"机器人的激光传感器安全吗？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[3],"reference_answer":"安全，符合IEC 60825-1:2014的1类激光产品标准，不会产生危险的激光辐射。","reference_contexts":["本产品激光测距传感器符合 IEC 60825-1:2014 的 1 类激光产品标准，不会产生危险的激光辐射。"]},
    {"question":"震动拖布支架可以拆卸吗？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[7],"reference_answer":"不可以，震动拖布支架不可拆卸。","reference_contexts":["提示：震动拖布支架不可拆卸。"]},
    # ═══ Installation (p9-p11) ═══
    {"question":"安装基座前需要移除什么？","question_type":"setup","difficulty":"easy","modality_required":"text","gold_pages":[10],"reference_answer":"需要取出底部高速自清洁刷组件的运输固定泡沫。","reference_contexts":["将基座主体放置在硬质水平地面上，取出底部高速自清洁刷组件的运输固定泡沫。"]},
    {"question":"基座底板的安装到位标志是什么？","question_type":"setup","difficulty":"easy","modality_required":"text","gold_pages":[10],"reference_answer":"听到'咔哒'声后表示基座底板和基座主体连接到位。","reference_contexts":["在听到'咔哒'声后，表示基座底板和基座主体连接到位。"]},
    {"question":"首次清扫时应该注意什么？","question_type":"setup","difficulty":"medium","modality_required":"text","gold_pages":[9,13],"reference_answer":"首次清扫建议全程跟随主机，协助处理可能存在的小问题。","reference_contexts":["首次清扫过程中建议全程跟随主机，协助处理一些可能存在的小问题","首次扫拖时，请跟随并帮助机器人排除一些不友好的小问题。"]},
    {"question":"是否支持 5GHz WiFi？","question_type":"setup","difficulty":"easy","modality_required":"text","gold_pages":[12],"reference_answer":"不支持，WiFi仅支持2.4GHz频段。","reference_contexts":["WiFi 无线连接仅支持 2.4GHz 频段的网络，暂不支持 5GHz 频段。"]},
    {"question":"机器人支持哪两款APP？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[11],"reference_answer":"支持石头APP和米家APP。","reference_contexts":["方式1：在应用商店搜索Roborock或扫描下方二维码下载安装石头APP","方式2：在应用商店搜索米家或扫描下方二维码下载安装米家APP","本产品支持石头 APP 和米家 APP 操控"]},
    # ═══ Operation (p12-p14) ═══
    {"question":"机器人充电时可以关机吗？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[12],"reference_answer":"不可以，机器人在充电中无法关机。","reference_contexts":["提示：机器人在充电中无法关机。"]},
    {"question":"勿扰模式的默认时间段是什么？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[13],"reference_answer":"22:00至08:00，可通过APP修改。","reference_contexts":["出厂默认开启时间段为 22:00-08:00，可使用手机 APP 修改勿扰时间段或关闭。"]},
    {"question":"局部清扫的范围是多大？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[13],"reference_answer":"以机器人为中心1.5米×1.5米的方形区域。","reference_contexts":["清扫范围：机器人为中心 1.5 米 ×1.5 米的方形区域。"]},
    {"question":"机器人休眠多久后自动关机？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[13],"reference_answer":"休眠时间超过12小时自动关机。","reference_contexts":["休眠时间超过 12 小时自动关机。"]},
    {"question":"APP的个性清洁方式包括哪些？","question_type":"feature","difficulty":"hard","modality_required":"text","gold_pages":[14],"reference_answer":"定时清洁、选区清洁、划区清洁、指哪到哪、遥控模式、禁区虚拟墙、定制顺序、地毯清洁设置、少碰撞模式。","reference_contexts":["定时清洁｜选区清洁｜划区清洁｜指哪到哪｜遥控模式｜禁区虚拟墙｜定制顺序｜地毯清洁设置｜少碰撞模式"]},
    {"question":"清洁模式可以调节哪些参数？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[14],"reference_answer":"扫拖方式、清扫吸力、擦地强度、拖地偏好、洗布模式、集尘模式。","reference_contexts":["清洁模式调节：扫拖方式｜清扫吸力｜擦地强度｜拖地偏好｜洗布模式｜集尘模式"]},
    {"question":"重置系统后哪些设置会丢失？","question_type":"troubleshooting","difficulty":"medium","modality_required":"text","gold_pages":[14],"reference_answer":"定时清扫和WiFi等相关设置将恢复到出厂状态。","reference_contexts":["提示：重置系统后定时清扫和 WiFi 等相关设置将恢复到出厂状态。"]},
    # ═══ Maintenance (p15-p21) ═══
    {"question":"软胶主刷的建议更换频率是多久？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[15],"reference_answer":"每6-12个月更换。","reference_contexts":["软胶主刷：每2周清理，每6-12个月更换。"]},
    {"question":"边刷需要多久清理一次？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[15],"reference_answer":"每1个月清理一次。","reference_contexts":["边刷：每1个月清理，每3-6个月更换。"]},
    {"question":"一次性尘袋需要多久更换？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[15],"reference_answer":"按需更换，一般每1-2个月。","reference_contexts":["一次性尘袋：按需更换，每1-2个月。"]},
    {"question":"银离子抑菌模块需要多久更换？","question_type":"maintenance","difficulty":"medium","modality_required":"text","gold_pages":[15],"reference_answer":"每12个月更换。","reference_contexts":["银离子抑菌模块：按需更换，每12个月。"]},
    {"question":"万向轮如何拆卸清理？","question_type":"maintenance","difficulty":"medium","modality_required":"text","gold_pages":[16],"reference_answer":"借助小螺丝刀等工具撬出轮轴，拔出轮体，冲洗轮体和轮轴上的毛发或脏物，晾干后装回。","reference_contexts":["借助小螺丝刀等工具撬出轮轴，拔出轮体","冲洗轮体和轮轴上的毛发或脏物，晾干后装回轮体并压紧","提示：万向轮支架不可取出。"]},
    {"question":"滤网可以用刷子清理吗？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[17],"reference_answer":"不可以，禁止用手、刷子或者尖锐物触碰滤网表面，以免损伤滤网。","reference_contexts":["提示：禁止用手、刷子或者尖锐物触碰滤网表面，以免损伤滤网。"]},
    {"question":"哪些传感器需要每1个月清理？","question_type":"maintenance","difficulty":"medium","modality_required":"text","gold_pages":[15],"reference_answer":"回充传感器、悬崖传感器、沿墙传感器、地毯识别传感器、充电触片、主轮、基座充电弹片及信号发射区。","reference_contexts":["回充传感器：每1个月清理","悬崖传感器：每1个月清理","沿墙传感器：每1个月清理","地毯识别传感器：每1个月清理","充电触片：每1个月清理","主轮：每1个月清理","基座充电弹片、信号发射区及机身：每1个月清理"]},
    {"question":"清洁尘盒时可以用洗涤剂吗？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[17],"reference_answer":"不可以，请用清水清洗，不要添加任何洗涤剂，否则可能造成滤网堵塞。","reference_contexts":["提示：请用清水清洗，不要添加任何洗涤剂，否则可能造成滤网堵塞。"]},
    {"question":"如何搬动基座？","question_type":"maintenance","difficulty":"medium","modality_required":"text","gold_pages":[18],"reference_answer":"一只手抓紧背面扣手，另一只手抓紧正面内侧向上抬起。请勿直接抬起基座底板搬运。","reference_contexts":["请一只手抓紧背面扣手，另一只手抓紧正面内侧，如图所示向上抬起搬动。请勿直接抬起基座底板搬运，避免发生基座主体砸落危险。"]},
    {"question":"高速自清洁刷卡扣如何操作？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[19],"reference_answer":"向上提起卡扣取出清洁刷，清除缠绕物并冲洗干净后装回原位，扣上卡扣确保扣合到位。","reference_contexts":["向上提起高速自清洁刷卡扣，取出高速自清洁刷","清除高速自清洁刷缠绕物并冲洗干净后装回原位，扣上卡扣并确保扣合到位"]},
    {"question":"清洗槽滤网安装到位的标志是什么？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[20],"reference_answer":"听到'咔哒'声表示安装到位。","reference_contexts":["请确认听到'咔哒'声，以确保安装到位。"]},
    {"question":"更换一次性尘袋时应注意什么？","question_type":"maintenance","difficulty":"medium","modality_required":"text","gold_pages":[20],"reference_answer":"取出时尘袋提手会封闭袋口防止灰尘漏出，新尘袋要插入卡槽确保安装到位，封口要牢靠。","reference_contexts":["取出时，尘袋提手会将尘袋封闭起来，有效防止灰尘漏出","按图示将新的一次性尘袋插入卡槽，并确保安装到位","请确保一次性尘袋封口牢靠，避免垃圾泄露损坏基座"]},
    {"question":"风道堵塞时如何清理？","question_type":"maintenance","difficulty":"hard","modality_required":"text","gold_pages":[21],"reference_answer":"取出净水箱、污水箱、集尘桶，将基座稳固放在垫好毛巾的硬质地面上，用螺丝刀拧下风道盖板8颗螺丝，清理擦拭后装回并拧紧螺丝。","reference_contexts":["取出净水箱、污水箱、集尘桶","将基座稳固放置在提前垫好柔软毛巾的硬质地面上","用螺丝刀拧下风道盖板8颗螺丝并取下风道盖板","用干抹布清理及擦拭风道及风道盖板，清理干净后装回风道盖板并拧紧风道盖板8颗螺丝"]},
    {"question":"拖布自清洁效果不佳怎么办？","question_type":"troubleshooting","difficulty":"medium","modality_required":"text","gold_pages":[24],"reference_answer":"拖布未粘贴平整请重新粘贴；环境较脏建议使用APP将洗布模式切换为深度洗。","reference_contexts":["拖布未粘贴平整，请重新粘贴平整；环境较脏，建议使用手机APP 将洗布模式切换为'深度洗'提升清洁效果。"]},
    # ═══ Troubleshooting (p24) ═══
    {"question":"为什么没有自动集尘？","question_type":"troubleshooting","difficulty":"medium","modality_required":"text","gold_pages":[24],"reference_answer":"可能原因：APP关闭了自动集尘、集尘桶未安装、机器人未清扫过、勿扰时间段内。","reference_contexts":["APP 关闭了自动集尘功能，请检查 APP 设置；集尘桶未安装；机器人若未清扫过，自动返回基座不集尘；勿扰时间段内机器人自动返回基座将不会主动集尘"]},
    {"question":"为什么没有洗拖布？","question_type":"troubleshooting","difficulty":"medium","modality_required":"text","gold_pages":[24],"reference_answer":"可能原因：机器人未拖过地、未从基座出发、净水箱无水或未安装、污水箱水满或未安装、清洗槽滤网未安装到位。","reference_contexts":["机器人若未拖过地，将不会主动清洗拖布；若机器人没有从基座出发，App 地图上没有基座；净水箱无水或未安装，污水箱水满或未安装；清洗槽滤网未安装或未安装到位。"]},
    {"question":"基座LED红色常亮是什么原因？","question_type":"troubleshooting","difficulty":"hard","modality_required":"text","gold_pages":[24],"reference_answer":"集尘桶或尘袋不在位、电压异常、净水箱无水或未安装、污水箱水满或未安装、清洗槽滤网未安装到位。","reference_contexts":["基座LED状态指示灯红色长亮：集尘桶或尘袋不在位；电压异常；净水箱无水或未安装；污水箱水满或未安装；清洗槽滤网未安装或未'咔哒'安装到位"]},
    {"question":"错误42是什么原因？","question_type":"troubleshooting","difficulty":"hard","modality_required":"text","gold_pages":[24],"reference_answer":"高速自清洁刷模组停靠在清洗槽左侧或右侧，可能卡住异物或清洗槽滤网未按压到底。","reference_contexts":["机器人语音播报：错误42 且基座高速自清洁刷模组停靠在清洗槽左侧或右侧。停靠在左侧请检查清洗槽左侧是否有异物卡住；停靠在右侧清洗槽滤网旁，可能滤网上有异物卡住或清洗槽滤网未'咔哒'按压到底"]},
    {"question":"机器人突然漏扫是什么原因？","question_type":"troubleshooting","difficulty":"medium","modality_required":"text","gold_pages":[24],"reference_answer":"沿墙传感器、悬崖传感器或地毯识别传感器可能脏污，建议用柔软干布擦拭。","reference_contexts":["怀疑沿墙传感器、悬崖传感器或地毯识别传感器已脏污，建议使用柔软干布擦拭干净。"]},
    {"question":"前三次使用需要充电十六小时吗？","question_type":"troubleshooting","difficulty":"easy","modality_required":"text","gold_pages":[24],"reference_answer":"不需要，锂离子电池随用随充无记忆效应，充满即用。","reference_contexts":["锂离子电池随用随充无记忆效应，充满即用无须等待十六小时。"]},
    {"question":"清扫中途电量不足回充但未续扫怎么办？","question_type":"troubleshooting","difficulty":"medium","modality_required":"text","gold_pages":[24],"reference_answer":"确认机器人未处于勿扰模式下（该模式下不会续扫）；手动回充也不会续扫。","reference_contexts":["请确认机器人未处于勿扰模式下，该模式下不会续扫；手动回充或将机器人放回基座不会续扫。"]},
    # ═══ Specs / Battery (p22-p23) ═══
    {"question":"机器人的电池规格是什么？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[22],"reference_answer":"14.4V/5200mAh锂离子电池。","reference_contexts":["电池：14.4V/5200mAh 锂离子电池"]},
    {"question":"机器人的额定功率是多少？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[22],"reference_answer":"69W。","reference_contexts":["额定功率：69W"]},
    {"question":"充电时间需要多久？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[22],"reference_answer":"小于6小时。","reference_contexts":["充电时间：＜6小时"]},
    {"question":"机器人的重量是多少？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[22],"reference_answer":"约4.7kg。","reference_contexts":["产品重量：约4.7kg"]},
    {"question":"基座的额定输入电压范围是多少？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[22],"reference_answer":"220V-240V~50-60Hz。","reference_contexts":["额定输入：220V-240V～50-60Hz"]},
    {"question":"集尘状态的功率是多少？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[22],"reference_answer":"1000W。","reference_contexts":["功率（集尘状态）：1000W"]},
    # ═══ Warranty (p25-p26) ═══
    {"question":"七天无理由退货的条件是什么？","question_type":"warranty","difficulty":"medium","modality_required":"text","gold_pages":[25],"reference_answer":"自签收次日起7日内，产品（含包装和附件）完好不影响二次销售前提下可申请，运费由用户承担。","reference_contexts":["自签收次日起 7 日内，在保证产品（含包装和附件）完好不影响二次销售前提下，可申请无理由退货，运费由用户自行承担"]},
    {"question":"十五天换货的条件是什么？","question_type":"warranty","difficulty":"medium","modality_required":"text","gold_pages":[25],"reference_answer":"自签收次日起15日内，出现产品性能故障表所列情况，经检测确认后可免费换货，运费由石头科技承担。","reference_contexts":["自签收次日起15日内，本产品出现《石头自清洁集尘充电座产品性能故障表》所列情况，经由石头科技售后服务中心检测确定为产品性能故障问题，可免费享受换货服务"]},
    {"question":"哪些耗材无保修期？","question_type":"warranty","difficulty":"medium","modality_required":"text","gold_pages":[25],"reference_answer":"软胶主刷、边刷、滤网、高速自清洁刷等随机耗材无保修期。","reference_contexts":["注：随机耗材无保修期（软胶主刷、边刷、滤网、高速自清洁刷等）。"]},
    {"question":"哪些情况不能享受免费保修？","question_type":"warranty","difficulty":"hard","modality_required":"text","gold_pages":[25],"reference_answer":"未经授权维修拆机、误用碰撞疏忽进液、正常磨损、不可抗力、超保修期、不符合性能故障表所列情况。","reference_contexts":["未经石头科技授权的机构、人员对产品进行了维修、检验、拆机等操作","未按照本产品说明书使用，导致误用、碰撞、疏忽、进液、事故、改动","正常的磨损","因不可抗力造成的损坏","已超过保修有效期限","不符合《石头自清洁集尘充电座产品性能故障表》所列的情况"]},
    {"question":"产品的售后服务电话是多少？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[23,27],"reference_answer":"400-900-1755。","reference_contexts":["可拨打石头科技服务电话：400-900-1755","服务电话：400-900-1755"]},
    # ═══ Battery disposal (p23) ═══
    {"question":"报废前如何取出电池？","question_type":"maintenance","difficulty":"hard","modality_required":"text","gold_pages":[23],"reference_answer":"在不接触基座的情况下运行至低电量后关机，卸下电池盖板螺丝，取下盖板，按下卡扣拔出连接器插头并取下电池。","reference_contexts":["让机器人在不接触基座的情况下运行至无法清扫的低电量状态","将机器人关机","将机器人电池盖板螺丝卸下","取下电池盖板","按下卡扣拔出电池的连接器插头并取下电池"]},
    {"question":"异常状态下机器人会怎样？","question_type":"troubleshooting","difficulty":"easy","modality_required":"text","gold_pages":[23],"reference_answer":"电源指示灯红色快闪并语音提示，10分钟无操作后自动休眠。","reference_contexts":["机器人电源指示灯红色快闪并语音提示，请按照语音及APP提示解决异常","异常状态下 10 分钟无操作，机器人自动休眠"]},
    # ═══ Cross-page / Inference ═══
    {"question":"如何判断机器人是电量不足还是电池温度异常？","question_type":"troubleshooting","difficulty":"hard","modality_required":"text","gold_pages":[24,5],"reference_answer":"电量不足时电源指示灯红色（<20%），电池温度异常时也会无法开机。可先充电尝试，若充电后仍无法开机可能是温度问题，需在0-40℃环境使用。","reference_contexts":["无法开机：电池电量不足，请先靠上基座充电后再使用；电池温度过低或过高，请在0-40℃环境下使用","红色：电量＜20%"]},
    {"question":"从开箱到完成首次自动清扫的完整流程是什么？","question_type":"setup","difficulty":"hard","modality_required":"text","gold_pages":[10,11,12,13],"reference_answer":"1.组装基座（取出固定泡沫，连接底板，插电源线）2.靠墙放置基座（两侧0.5m前方1.5m空间）3.下载APP并连接WiFi（仅2.4GHz）4.机器人靠上基座充电 5.短按清扫键启动扫拖。","reference_contexts":["将基座主体放置在硬质水平地面上，取出底部高速自清洁刷组件的运输固定泡沫","在硬质水平地面靠墙放置基座，且两侧有0.5米、前方有1.5米、上方有1米以上空间","下载APP：方式1 Roborock/方式2 米家","重置WiFi进入配网模式","短按键机器人将按扫描生成的地图动态规划扫拖路线"]},
    {"question":"哪些行为会导致保修立即失效？","question_type":"warranty","difficulty":"hard","modality_required":"text","gold_pages":[3,25],"reference_answer":"禁止与任何类型的电源转换器一起使用，否则保修将立即失效；未经授权维修拆机也会导致不能免费保修。","reference_contexts":["禁止与任何类型的电源转换器一起使用，否则保修将立即失效","未经石头科技授权的机构、人员对产品进行了维修、检验、拆机等操作"]},
    {"question":"为什么机器人有时停在基座旁边但无法对接充电？","question_type":"troubleshooting","difficulty":"hard","modality_required":"text","gold_pages":[24,11],"reference_answer":"可能原因：基座附近障碍物太多、机器人距离基座太远、充电触片脏污、基座信号发射区被遮挡。","reference_contexts":["无法回充：基座附近障碍物太多，请将基座放到开阔区域；机器人距离基座太远","接触不良，请清理基座充电弹片与机器人充电触片","请勿将基座放置于阳光直晒的地方或用任何物体遮挡信号发射区"]},
    {"question":"扫拖过程中断后能自动续扫吗？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[13],"reference_answer":"能。电量不足自动回充后可自动回到中断处继续扫拖，但勿扰模式下不会续扫。","reference_contexts":["扫拖过程中电量不足时，机器人将自动返回基座充电，电量充足后自动回到中断处继续扫拖","请确认机器人未处于勿扰模式下，该模式下不会续扫"]},
    {"question":"机器人在工作过程中可以移动基座吗？","question_type":"setup","difficulty":"easy","modality_required":"text","gold_pages":[13],"reference_answer":"不可以，扫拖过程中请勿移动基座。","reference_contexts":["扫拖过程中，请勿移动基座。"]},
    {"question":"机器人能同时扫地和拖地吗？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[13,14],"reference_answer":"可以，扫拖方式可在APP中调节，包括仅扫地、仅拖地、扫拖一体。","reference_contexts":["扫拖方式｜清扫吸力｜擦地强度｜拖地偏好｜洗布模式｜集尘模式","机器人将默认收起拖布支架清扫地毯"]},
    # ═══ Additional specific content questions ═══
    {"question":"清水箱每次加水可以加多少清洁液？","question_type":"maintenance","difficulty":"medium","modality_required":"text","gold_pages":[12],"reference_answer":"可选加入四分之三盖（约15毫升）石头官方清洁液，加水量勿超过Max水位线。","reference_contexts":["可选加入四分之三盖（约15毫升）石头官方清洁液，再加入自来水，加水量请勿超过Max水位线"]},
    {"question":"机器人WiFi等待连接超时后会怎样？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[12],"reference_answer":"等待连接状态超过1小时后，将自动关闭WiFi功能，需重置WiFi后重新联网。","reference_contexts":["机器人处于等待连接状态超过 1 小时后，将自动关闭 WiFi 功能。如需再次连接，请重置 WiFi 后联网。"]},
    {"question":"清洁结束后机器人会自动做什么？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[14],"reference_answer":"自动返回基座充电，并根据历史清洁行为判断是否需要洗拖布和集尘。","reference_contexts":["清洁结束后，机器人会自动返回基座充电，并根据历史清洁行为判断本次返回基座后是否需要洗拖布和集尘。"]},
    {"question":"为什么基座出厂测试后水路有残留水？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[11],"reference_answer":"每一台基座出厂前都会进行有水测试，基站水路残留少量水属于正常现象。","reference_contexts":["每一台基座出厂前都会进行有水测试，基站水路残留少量水属于正常现象。"]},
    {"question":"滤网建议如何轮换使用？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[18],"reference_answer":"建议两个滤网轮换使用。","reference_contexts":["建议两个滤网轮换使用。"]},
    {"question":"如何在APP中设置软件禁区？","question_type":"setup","difficulty":"medium","modality_required":"text","gold_pages":[9,14],"reference_answer":"在手机APP中设置禁区虚拟墙，阻挡主机进入不需要清扫的区域。","reference_contexts":["如果某些区域不需要清扫或可能卡住主机，可在手机 APP 中设置软件禁区","禁区虚拟墙"]},
    {"question":"机器人充电时电源指示灯什么状态？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[14],"reference_answer":"电源指示灯呼吸闪烁。","reference_contexts":["充电时，电源指示灯呼吸闪烁。"]},
    {"question":"扫拖时间少于多少分钟会默认扫拖两次？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[13],"reference_answer":"少于10分钟将默认扫拖两次。","reference_contexts":["扫拖时间小于 10 分钟，将默认扫拖两次。"]},
    {"question":"主刷脏污应该怎么清理？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[16],"reference_answer":"建议用湿布擦拭。如果浸水，务必晾干防止暴晒。","reference_contexts":["主刷脏污建议用湿布擦拭。如果浸水，务必晾干防止暴晒。"]},
    {"question":"运输机器人时应该注意什么？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[3],"reference_answer":"确保处于断电关机状态，建议使用原包装盒包装。","reference_contexts":["如需运输产品，请确保处于断电、关机状态，并建议使用原包装盒包装。"]},
    {"question":"可以在机器人工作时靠近刷头吗？","question_type":"safety","difficulty":"easy","modality_required":"text","gold_pages":[3],"reference_answer":"不可以，请勿在机器人工作时让人或宠物的肢体等部位靠近刷头旋转位置。","reference_contexts":["请勿在机器人工作时，让人或宠物的肢体等部位靠近刷头旋转位置，避免产生伤害。"]},
    {"question":"暂停状态下将机器人靠上基座充电会怎样？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[13],"reference_answer":"将结束本次清扫。","reference_contexts":["提示：暂停时将机器人靠上基座充电，将结束本次清扫。"]},
    {"question":"充电接触区域有脏污会有什么影响？","question_type":"troubleshooting","difficulty":"easy","modality_required":"text","gold_pages":[24],"reference_answer":"会导致充电速度慢，请用干布清理该区域。","reference_contexts":["充电接触区域可能有脏污，请用干布清理该区域。"]},
    {"question":"基座可以在非硬质地面上放置吗？","question_type":"setup","difficulty":"easy","modality_required":"text","gold_pages":[11],"reference_answer":"不可以，在非硬质地面（如地毯、地垫等）上摆放基座有倾倒风险，且机器人可能无法出桩。","reference_contexts":["在非硬质地面（如地毯，地垫等）上摆放基座有倾倒风险，且机器人可能无法出桩。"]},
    {"question":"智能音箱可以控制机器人吗？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[14],"reference_answer":"可以，APP支持智能音箱语控功能。","reference_contexts":["更多特色功能：音量调节｜个性化语音｜勿扰模式｜智能音箱语控等更多功能"]},
    {"question":"边刷如何拆卸？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[16],"reference_answer":"拆下边刷固定螺丝，清理后装回并锁紧螺丝。","reference_contexts":["拆下边刷固定螺丝","清理边刷后装回并锁紧螺丝"]},
    {"question":"软胶主刷罩安装到位的标准是什么？","question_type":"maintenance","difficulty":"medium","modality_required":"text","gold_pages":[16],"reference_answer":"确保4齿插入对应卡槽中，4齿未露出即代表安装到位。","reference_contexts":["确保软胶主刷罩 4 齿插入对应卡槽中，4 齿未露出即代表安装到位"]},
    {"question":"清水箱加水时水温有什么限制？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[12,19],"reference_answer":"请勿装入热水，可能造成水箱变形。","reference_contexts":["请勿装入热水，可能造成水箱变形。"]},
    {"question":"集尘效果差可能是什么原因？","question_type":"troubleshooting","difficulty":"hard","modality_required":"text","gold_pages":[24],"reference_answer":"主刷或主刷罩未安装到位、滤网/风道/集尘口/集尘进风口/尘盒/尘袋堵塞。","reference_contexts":["机器人主刷、主刷罩未安装到位，请重新安装并确保安装到位；滤网、风道、集尘口、集尘进风口、尘盒、尘袋堵塞，请清理。"]},
    {"question":"勿扰模式下机器人有什么行为变化？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[13],"reference_answer":"不自动续扫、按键灯亮度减弱、语音播报音量降低。","reference_contexts":["勿扰时间段内，机器人不自动续扫，按键灯亮度减弱，语音播报音量降低。"]},
    {"question":"产品保修卡建议保留包装箱多久？","question_type":"warranty","difficulty":"easy","modality_required":"text","gold_pages":[26],"reference_answer":"建议自签收之日起至少保留包装箱30天。","reference_contexts":["因运输过程中需使用包装箱保证产品运输安全，建议您自签收之日起至少保留包装箱30 天。"]},
    {"question":"机器人可以清扫装修废料吗？","question_type":"troubleshooting","difficulty":"easy","modality_required":"text","gold_pages":[9],"reference_answer":"不可以，请勿让机器人吸取硬物或尖锐物体（如装修废料、玻璃、铁钉等），否则可能会划伤机器人及地面。","reference_contexts":["请勿让扫拖机器人吸取硬物或尖锐物体（如装修废料，玻璃，铁钉等），否则可能会划伤机器人及地面。"]},
    {"question":"如果电池有渗出物接触到皮肤怎么办？","question_type":"safety","difficulty":"medium","modality_required":"text","gold_pages":[23],"reference_answer":"请用大量清水冲洗并及时就医。","reference_contexts":["如果电池有渗出物并不慎接触到，请用大量清水冲洗并及时就医。"]},
    {"question":"软胶主刷盖如何取下？","question_type":"maintenance","difficulty":"medium","modality_required":"text","gold_pages":[16],"reference_answer":"按解锁标志方向旋转并取下。","reference_contexts":["按解锁标志方向旋转并取下软胶主刷盖"]},
    {"question":"机器人进水了怎么办？","question_type":"troubleshooting","difficulty":"medium","modality_required":"text","gold_pages":[3],"reference_answer":"请确保机器人工作的地面无积水，否则可能会造成主机进水损坏。如已进水，请关机断电后联系客服。","reference_contexts":["请确保机器人工作的地面无积水，否则可能会造成主机进水损坏"]},
    {"question":"基座未安装尘袋时可以使用吗？","question_type":"maintenance","difficulty":"easy","modality_required":"text","gold_pages":[20],"reference_answer":"尘袋未安装时请勿装上集尘桶，可避免无尘袋自动集尘，也可通过APP关闭自动集尘功能。","reference_contexts":["尘袋未安装时请勿装上集尘桶，可避免无尘袋自动集尘，或者可通过手机 APP 关闭自动集尘功能。"]},
    {"question":"中国港澳台地区可以使用APP全部功能吗？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[3,26],"reference_answer":"不可以，产品仅限中国大陆地区销售和使用，在港澳台及海外使用将无法体验APP及智能音箱远程控制功能。","reference_contexts":["本产品仅限中国大陆地区销售和使用，在港澳台及海外使用将无法体验 APP 及智能音箱远程控制功能","本产品仅限中国大陆地区（不含港澳台地区）销售"]},
    # ═══ More multi-step/inference questions ═══
    {"question":"机器人的清扫路线规划是什么顺序？","question_type":"feature","difficulty":"medium","modality_required":"text","gold_pages":[13],"reference_answer":"先沿墙后Z字形填充的方式逐一完成各分区的扫拖。","reference_contexts":["采用先沿墙后 Z 字形填充的方式逐一完成各分区的扫拖"]},
    {"question":"机器人未找到基座时会怎样？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[14],"reference_answer":"将自动返回起点位置，需手动放回基座充电。","reference_contexts":["如机器人未找到基座将自动返回起点位置，请手动将机器人放回基座充电。"]},
    {"question":"什么情况下机器人电源指示灯会红色快闪？","question_type":"troubleshooting","difficulty":"easy","modality_required":"text","gold_pages":[5,23],"reference_answer":"异常状态时红色快闪并语音提示。","reference_contexts":["红色快闪：异常状态","机器人电源指示灯红色快闪并语音提示"]},
    {"question":"机器人支持语音控制吗？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[14],"reference_answer":"支持，通过智能音箱可以实现语控，如音量调节、个性化语音等。","reference_contexts":["更多特色功能：音量调节｜个性化语音｜勿扰模式｜智能音箱语控等更多功能"]},
    {"question":"说明书中提到的执行标准有哪些？","question_type":"feature","difficulty":"hard","modality_required":"text","gold_pages":[3],"reference_answer":"GB 4706.1-2005, GB 4706.7-2014, GB 4343.1-2018, GB 17625.1-2012。","reference_contexts":["执行标准：GB 4706.1-2005, GB 4706.7-2014, GB 4343.1-2018, GB 17625.1-2012"]},
    {"question":"制造商是哪家公司？","question_type":"feature","difficulty":"easy","modality_required":"text","gold_pages":[27],"reference_answer":"北京石头世纪科技股份有限公司。","reference_contexts":["制造商：北京石头世纪科技股份有限公司"]},
]

# Expected format: {question, question_type, difficulty, modality_required, gold_pages, reference_answer, reference_contexts, source_document, review_status}


def main() -> None:
    # Load existing 20
    existing_path = PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json"
    with open(existing_path, encoding="utf-8") as f:
        existing = json.load(f)

    # Normalize existing: add reference_contexts if missing (from reference_context)
    for q in existing:
        if "reference_contexts" not in q:
            rc = q.get("reference_context", "")
            q["reference_contexts"] = [rc] if rc else []
        q["source_document"] = q.get("source_document", "Roborock G10S")
        q["review_status"] = q.get("review_status", "ai_annotated")

    # Add new questions
    for q in NEW_QUESTIONS:
        q["source_document"] = "Roborock G10S"
        q["review_status"] = "ai_annotated"

    all_questions = existing + NEW_QUESTIONS

    # Save
    output_path = PROJECT_ROOT / "data" / "eval_dataset" / "golden_100.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    # Stats
    by_type = {}
    by_diff = {}
    by_mod = {}
    for q in all_questions:
        t = q.get("question_type", "?")
        d = q.get("difficulty", "?")
        m = q.get("modality_required", "text")
        by_type[t] = by_type.get(t, 0) + 1
        by_diff[d] = by_diff.get(d, 0) + 1
        by_mod[m] = by_mod.get(m, 0) + 1

    print(f"Golden Dataset: {len(all_questions)} questions")
    print(f"  By type: {by_type}")
    print(f"  By difficulty: {by_diff}")
    print(f"  By modality: {by_mod}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
