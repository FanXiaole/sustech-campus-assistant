"""
=============================================================================
SUSTech Test Set Builder — 50+ 结构化评测问题集
=============================================================================
构建一个多样化的测试集，覆盖简单事实到复杂推理的不同难度层级。

设计原则：
  1. 覆盖所有难度层级（easy/medium/hard/oos）
  2. 覆盖所有查询类型（factual_simple, procedural, comparative 等）
  3. 每个问题标注 ground_truth + key_facts + source_urls
  4. 包含"范围外"问题来测试 abstention 机制

问题类别与分布（50 题）：
  easy (15):   factual_simple(8) + time_location(4) + procedure(3)
  medium (20): factual_complex(8) + department_info(7) + policy(5)
  hard (10):   comparative(4) + cross_source(4) + temporal(2)
  oos (5):     external(3) + hallucination_bait(2)

使用方法：python evaluation/test_set_builder.py
=============================================================================
"""

import json
from pathlib import Path

# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

OUTPUT_PATH = DATA_DIR / "test_set.json"


def build_test_set() -> list[dict]:
    """
    构建完整的 50 题测试集。

    注意：这里的 ground_truth 和 key_facts 是"预期答案框架"。
    实际评分时，LLM 的回答需要与这些关键事实比对，而不是逐字匹配。
    因为同一个事实可以用不同方式表述。
    """

    test_set = []

    # ======================================================================
    # EASY — 15 题：单跳事实，答案在单一 chunk 中
    # ======================================================================

    easy_factual = [
        {
            "q_id": "easy_factual_001",
            "question": "南方科技大学图书馆的工作日开放时间是什么？",
            "ground_truth": "周一至周五 8:00-22:00",
            "key_facts": ["8:00", "22:00", "周一至周五"],
            "source_urls": ["https://lib.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "factual_simple",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_factual_002",
            "question": "南科大计算机科学与工程系是哪一年成立的？",
            "ground_truth": "2011年（或建校之初即设立）",
            "key_facts": ["计算机科学与工程系", "2011"],
            "source_urls": ["https://cse.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "factual_simple",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_factual_003",
            "question": "南方科技大学的英文名称是什么？",
            "ground_truth": "Southern University of Science and Technology，简称SUSTech",
            "key_facts": ["Southern University of Science and Technology", "SUSTech"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "factual_simple",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_factual_004",
            "question": "南科大有多少个书院？",
            "ground_truth": "6个书院：致仁、树仁、致诚、树德、致新、树礼",
            "key_facts": ["6", "致仁", "树仁", "致诚", "树德", "致新", "树礼"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "factual_simple",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_factual_005",
            "question": "南科大的校训是什么？",
            "ground_truth": "明德求是 日新自强",
            "key_facts": ["明德求是", "日新自强"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "factual_simple",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_factual_006",
            "question": "南方科技大学校园在深圳哪个区？",
            "ground_truth": "南山区",
            "key_facts": ["南山区", "深圳"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "factual_simple",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_factual_007",
            "question": "南科大是一所什么类型的高校？",
            "ground_truth": "公办新型研究型大学，理工科为主",
            "key_facts": ["公办", "研究型", "理工科"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "factual_simple",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_factual_008",
            "question": "致仁书院的名字有什么寓意？",
            "ground_truth": "取自校名'南方科技大学'中蕴含的格物致知精神",
            "key_facts": ["致仁", "格物致知"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "factual_simple",
            "expected_abstain": False,
        },
    ]

    easy_time_location = [
        {
            "q_id": "easy_time_001",
            "question": "南科大图书馆周末的开放时间是几点到几点？",
            "ground_truth": "周末 9:00-21:00",
            "key_facts": ["周末", "9:00", "21:00"],
            "source_urls": ["https://lib.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "time_location",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_time_002",
            "question": "南方科技大学的具体地址是什么？",
            "ground_truth": "广东省深圳市南山区学苑大道1088号",
            "key_facts": ["深圳市", "南山区", "学苑大道", "1088"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "time_location",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_time_003",
            "question": "南科大校历中，春季学期通常什么时候开学？",
            "ground_truth": "通常在2月中下旬",
            "key_facts": ["2月", "春季学期"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "time_location",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_time_004",
            "question": "南科大体育场在哪里？",
            "ground_truth": "校园内松禾体育场（或润杨体育馆）",
            "key_facts": ["体育场", "松禾"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "time_location",
            "expected_abstain": False,
        },
    ]

    easy_procedure = [
        {
            "q_id": "easy_proc_001",
            "question": "南科大新生如何办理校园卡？",
            "ground_truth": "入学时由学校统一发放，或在校园卡服务中心办理",
            "key_facts": ["校园卡", "入学", "统一发放", "服务中心"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "procedure",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_proc_002",
            "question": "如何在南科大图书馆借书？",
            "ground_truth": "凭校园卡在图书馆借阅台办理，或使用自助借还机",
            "key_facts": ["校园卡", "借书", "图书馆", "自助借还"],
            "source_urls": ["https://lib.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "procedure",
            "expected_abstain": False,
        },
        {
            "q_id": "easy_proc_003",
            "question": "南科大校园网如何连接？",
            "ground_truth": "连接SUSTech WiFi，使用CAS账号登录",
            "key_facts": ["WiFi", "CAS", "账号", "登录"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "easy",
            "category": "procedure",
            "expected_abstain": False,
        },
    ]

    # ======================================================================
    # MEDIUM — 20 题：需要综合多个 chunk 的信息
    # ======================================================================

    medium_complex = [
        {
            "q_id": "medium_complex_001",
            "question": "南科大计算机系有哪些主要的研究方向？请列举至少三个。",
            "ground_truth": "人工智能、数据科学、计算机系统、计算机网络、计算机视觉、自然语言处理等",
            "key_facts": ["人工智能", "数据科学", "计算机系统"],
            "source_urls": ["https://cse.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "factual_complex",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_complex_002",
            "question": "南方科技大学的本科生培养模式有什么特点？",
            "ground_truth": "1+3或2+2模式，前1-2年不分专业，通识教育+书院制",
            "key_facts": ["通识教育", "书院制", "不分专业"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "factual_complex",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_complex_003",
            "question": "南科大有哪些主要的奖学金类型？申请条件是什么？",
            "ground_truth": "新生奖学金、学业奖学金、国家奖学金、社会捐赠奖学金等",
            "key_facts": ["新生奖学金", "学业奖学金", "国家奖学金"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "factual_complex",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_complex_004",
            "question": "在南科大，本科生如何参与科研？有哪些途径？",
            "ground_truth": "通过导师制、大创项目、实验室开放日、暑期科研等途径",
            "key_facts": ["导师制", "大创项目", "实验室"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "factual_complex",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_complex_005",
            "question": "南科大有哪些国际合作交流项目？学生如何申请？",
            "ground_truth": "与多所国际高校有交换项目、暑期学校、联合培养等",
            "key_facts": ["国际交流", "交换", "联合培养"],
            "source_urls": ["https://ws.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "factual_complex",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_complex_006",
            "question": "南科大的校园交通工具有哪些？校车路线是什么？",
            "ground_truth": "校内有免费穿梭巴士，连接各书院、教学楼和校门",
            "key_facts": ["校车", "巴士", "穿梭"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "factual_complex",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_complex_007",
            "question": "南科大图书馆有哪些电子资源数据库？",
            "ground_truth": "包括CNKI、Web of Science、IEEE Xplore、Elsevier等",
            "key_facts": ["数据库", "CNKI", "Web of Science"],
            "source_urls": ["https://lib.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "factual_complex",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_complex_008",
            "question": "南科大的社团活动有哪些类型？如何加入？",
            "ground_truth": "学术类、文艺类、体育类、公益类等，通过社团招新加入",
            "key_facts": ["社团", "学术", "文艺", "体育", "招新"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "factual_complex",
            "expected_abstain": False,
        },
    ]

    medium_department = [
        {
            "q_id": "medium_dept_001",
            "question": "南科大数学系有哪些教授？他们的研究方向是什么？",
            "ground_truth": "需从数学系官网获取教师列表和研究方向",
            "key_facts": ["数学系", "教授", "研究方向"],
            "source_urls": ["https://math.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "department_info",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_dept_002",
            "question": "南科大物理系本科有哪些专业方向？",
            "ground_truth": "应用物理、物理学等方向",
            "key_facts": ["物理系", "本科", "专业方向"],
            "source_urls": ["https://phy.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "department_info",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_dept_003",
            "question": "南科大生物系有哪些实验室和科研平台？",
            "ground_truth": "包括多个省市级重点实验室",
            "key_facts": ["生物系", "实验室", "科研平台"],
            "source_urls": ["https://bio.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "department_info",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_dept_004",
            "question": "南科大金融系有哪些课程？培养方案包含什么内容？",
            "ground_truth": "包括经济学基础、金融学核心课程、量化金融等方向",
            "key_facts": ["金融系", "课程", "培养方案"],
            "source_urls": ["https://fin.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "department_info",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_dept_005",
            "question": "南科大材料科学与工程系的科研实力如何？有哪些代表性成果？",
            "ground_truth": "在新能源材料、电子信息材料等领域有突出成果",
            "key_facts": ["材料系", "科研", "成果"],
            "source_urls": ["https://mse.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "department_info",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_dept_006",
            "question": "南科大的医学院有哪些附属医院？",
            "ground_truth": "包括南科大第一附属医院、第二附属医院等",
            "key_facts": ["医学院", "附属医院"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "department_info",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_dept_007",
            "question": "南科大环境科学与工程学院的研究重点是什么？",
            "ground_truth": "水环境、大气环境、环境健康、可持续发展等",
            "key_facts": ["环境", "研究", "水资源", "大气"],
            "source_urls": ["https://env.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "department_info",
            "expected_abstain": False,
        },
    ]

    medium_policy = [
        {
            "q_id": "medium_policy_001",
            "question": "南科大的学术诚信政策是什么？抄袭会受到什么处罚？",
            "ground_truth": "违反学术诚信将受到警告、记过甚至开除等处分",
            "key_facts": ["学术诚信", "处分", "抄袭"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "policy",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_policy_002",
            "question": "南科大本科生转专业有什么条件和流程？",
            "ground_truth": "需满足学分和成绩要求，在规定时间内申请",
            "key_facts": ["转专业", "条件", "申请"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "policy",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_policy_003",
            "question": "南科大研究生招生的基本要求是什么？",
            "ground_truth": "需具备学士学位，通过全国统一考试或推免",
            "key_facts": ["研究生", "招生", "要求"],
            "source_urls": ["https://gs.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "policy",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_policy_004",
            "question": "南科大宿舍管理规定有哪些？访客可以进入宿舍吗？",
            "ground_truth": "宿舍有门禁管理，访客需登记",
            "key_facts": ["宿舍", "管理规定", "访客", "登记"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "policy",
            "expected_abstain": False,
        },
        {
            "q_id": "medium_policy_005",
            "question": "南科大关于休学和复学有什么具体规定？",
            "ground_truth": "需提交申请，经审批后可办理休学，在规定时间内申请复学",
            "key_facts": ["休学", "复学", "申请", "审批"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "medium",
            "category": "policy",
            "expected_abstain": False,
        },
    ]

    # ======================================================================
    # HARD — 10 题：需要多跳推理、跨来源整合
    # ======================================================================

    hard_comparative = [
        {
            "q_id": "hard_comp_001",
            "question": "南科大的计算机系和电子系在课程设置和科研方向上有什么主要区别？",
            "ground_truth": "计算机系侧重软件/算法/AI，电子系侧重硬件/芯片/通信",
            "key_facts": ["计算机系", "电子系", "区别", "软件", "硬件"],
            "source_urls": ["https://cse.sustech.edu.cn", "https://eee.sustech.edu.cn"],
            "difficulty": "hard",
            "category": "comparative",
            "expected_abstain": False,
        },
        {
            "q_id": "hard_comp_002",
            "question": "比较南科大书院制与传统大学的学院制在本科生管理上的差异。",
            "ground_truth": "书院制强调生活与教育的融合，学院负责学术、书院负责生活",
            "key_facts": ["书院制", "学院制", "差异", "学术", "生活"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "hard",
            "category": "comparative",
            "expected_abstain": False,
        },
        {
            "q_id": "hard_comp_003",
            "question": "南科大的数学系和统计与数据科学系有什么不同？哪个更适合想做AI的学生？",
            "ground_truth": "数学系侧重纯数学和理论基础，统计与数据科学系侧重应用统计和机器学习",
            "key_facts": ["数学系", "统计", "数据科学", "AI"],
            "source_urls": ["https://math.sustech.edu.cn"],
            "difficulty": "hard",
            "category": "comparative",
            "expected_abstain": False,
        },
        {
            "q_id": "hard_comp_004",
            "question": "南科大图书馆的纸质资源和电子资源在使用方式上各有什么优劣？",
            "ground_truth": "纸质书可借出但数量有限，电子资源可随时随地访问但需要校园网/VPN",
            "key_facts": ["纸质", "电子", "借阅", "VPN"],
            "source_urls": ["https://lib.sustech.edu.cn"],
            "difficulty": "hard",
            "category": "comparative",
            "expected_abstain": False,
        },
    ]

    hard_cross_source = [
        {
            "q_id": "hard_cross_001",
            "question": "从南科大官网和南科手册来看，新生入学后需要办理哪些手续？分别在哪些地方办理？",
            "ground_truth": "包括校园卡、银行卡、宿舍入住、体检、选课等，分别在校园卡中心、银行、宿舍、校医院、教务系统",
            "key_facts": ["校园卡", "银行卡", "宿舍", "体检", "选课"],
            "source_urls": ["https://www.sustech.edu.cn", "https://github.com/SUSTech-CRA/sustech-online-ng"],
            "difficulty": "hard",
            "category": "cross_source",
            "expected_abstain": False,
        },
        {
            "q_id": "hard_cross_002",
            "question": "南科大的校园餐饮选择有哪些？各食堂分别在什么位置？有什么特色？",
            "ground_truth": "包括学生食堂、教工食堂、荔园、欣园等，分布在校园不同区域",
            "key_facts": ["食堂", "荔园", "欣园", "餐饮"],
            "source_urls": ["https://www.sustech.edu.cn", "https://github.com/SUSTech-CRA/sustech-online-ng"],
            "difficulty": "hard",
            "category": "cross_source",
            "expected_abstain": False,
        },
        {
            "q_id": "hard_cross_003",
            "question": "南科大有哪些校内外交通方式？从学校到深圳北站怎么走最方便？",
            "ground_truth": "校内有穿梭巴士，校外有地铁5号线塘朗站、公交等",
            "key_facts": ["交通", "地铁", "塘朗", "深圳北站"],
            "source_urls": ["https://www.sustech.edu.cn", "https://github.com/SUSTech-CRA/sustech-online-ng"],
            "difficulty": "hard",
            "category": "cross_source",
            "expected_abstain": False,
        },
        {
            "q_id": "hard_cross_004",
            "question": "南科大在疫情期间有哪些特殊政策？这些政策从哪里获取最新信息？",
            "ground_truth": "防疫政策会动态调整，建议关注学校官网通知和学工部公告",
            "key_facts": ["防疫", "政策", "官网", "通知"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "hard",
            "category": "cross_source",
            "expected_abstain": False,
        },
    ]

    hard_temporal = [
        {
            "q_id": "hard_temp_001",
            "question": "南科大2026年最新的本科招生政策相比往年有什么变化？",
            "ground_truth": "需查阅最新的招生简章，对比往年的变化",
            "key_facts": ["2026", "招生", "变化"],
            "source_urls": ["https://admit.sustech.edu.cn"],
            "difficulty": "hard",
            "category": "temporal",
            "expected_abstain": False,
        },
        {
            "q_id": "hard_temp_002",
            "question": "最近一个学期南科大有哪些新增的课程或专业？",
            "ground_truth": "需查阅最新教务公告和院系新闻",
            "key_facts": ["新增", "课程", "专业", "学期"],
            "source_urls": ["https://www.sustech.edu.cn"],
            "difficulty": "hard",
            "category": "temporal",
            "expected_abstain": False,
        },
    ]

    # ======================================================================
    # OUT-OF-SCOPE — 5 题：测试 abstention 机制
    # ======================================================================

    oos_external = [
        {
            "q_id": "oos_ext_001",
            "question": "清华大学计算机系的培养方案是怎样的？",
            "ground_truth": "N/A — 与南科大无关",
            "key_facts": [],
            "source_urls": [],
            "difficulty": "oos",
            "category": "external",
            "expected_abstain": True,
        },
        {
            "q_id": "oos_ext_002",
            "question": "北京大学的图书馆开放到几点？",
            "ground_truth": "N/A — 与南科大无关",
            "key_facts": [],
            "source_urls": [],
            "difficulty": "oos",
            "category": "external",
            "expected_abstain": True,
        },
        {
            "q_id": "oos_ext_003",
            "question": "麻省理工学院的申请截止日期是什么时候？",
            "ground_truth": "N/A — 与南科大无关",
            "key_facts": [],
            "source_urls": [],
            "difficulty": "oos",
            "category": "external",
            "expected_abstain": True,
        },
    ]

    oos_hallucination_bait = [
        {
            "q_id": "oos_hall_001",
            "question": "南科大校长的私人手机号码是多少？",
            "ground_truth": "N/A — 不应被回答",
            "key_facts": [],
            "source_urls": [],
            "difficulty": "oos",
            "category": "hallucination_bait",
            "expected_abstain": True,
        },
        {
            "q_id": "oos_hall_002",
            "question": "南科大哪个食堂有米其林三星大厨？",
            "ground_truth": "N/A — 不存在的事实，不应被回答",
            "key_facts": [],
            "source_urls": [],
            "difficulty": "oos",
            "category": "hallucination_bait",
            "expected_abstain": True,
        },
    ]

    # ── 组装全部测试集 ──
    test_set = (
        easy_factual + easy_time_location + easy_procedure +
        medium_complex + medium_department + medium_policy +
        hard_comparative + hard_cross_source + hard_temporal +
        oos_external + oos_hallucination_bait
    )

    return test_set


def save_test_set(test_set: list[dict], output_path: Path = None):
    """保存测试集到 JSON 文件并打印统计。"""
    if output_path is None:
        output_path = OUTPUT_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    # ── 统计 ──
    print(f"\n{'='*60}")
    print(f"Test Set Builder — Complete")
    print(f"{'='*60}")
    print(f"Total questions: {len(test_set)}")

    difficulty_counts = {}
    category_counts = {}
    abstain_count = 0
    for q in test_set:
        d = q["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        c = q["category"]
        category_counts[c] = category_counts.get(c, 0) + 1
        if q["expected_abstain"]:
            abstain_count += 1

    print(f"\nBy difficulty:")
    for d in ["easy", "medium", "hard", "oos"]:
        print(f"  {d}: {difficulty_counts.get(d, 0)} questions")
    print(f"\nBy category:")
    for c, n in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n} questions")
    print(f"\nAbstention-expected questions: {abstain_count}")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_set = build_test_set()
    save_test_set(test_set)
