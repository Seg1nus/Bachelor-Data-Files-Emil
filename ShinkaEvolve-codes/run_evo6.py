#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path="evaluate.py")

strategy = "weighted"
if strategy == "uniform":
    # 1. Uniform from correct programs
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=0.0,
        exploitation_ratio=1.0,
    )
elif strategy == "hill_climbing":
    # 2. Hill Climbing (Always from the Best)
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=100.0,
        exploitation_ratio=1.0,
    )
elif strategy == "weighted":
    # 3. Weighted Prioritization
    parent_config = dict(
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
elif strategy == "power_law":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )
elif strategy == "power_law_high":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=2.0,
        exploitation_ratio=0.2,
    )
elif strategy == "beam_search":
    # 5. Beam Search
    parent_config = dict(
        parent_selection_strategy="beam_search",
        num_beams=10,
    )


db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    # Inspiration parameters
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    # Island migration parameters
    migration_interval=10,
    migration_rate=0.1,  # chance to migrate program to random island
    island_elitism=True,  # Island elite is protected from migration
    **parent_config,
)

search_task_sys_msg = """You are an expert mathematician specializing in combinatoric problems.

Your task is to write the function that predicts the integers in a set, depending on a natural number, L.
The following are the expression for the set for L=1,2,3,4,5:
set(1)={-2}

set(2)={8,16}

set(3)={-384,-96,-56,-24,-16,-8,-4,4,8,16,24,28}

set(4)={-1248,-1216,-1152,-768,-480,-448,-384,-352,-304,-288,-264,-256,-240,-224,-192,-160,-144,-128,-112,-96,-88,-80,-72,-64,-48,-40,-32,-16,-8,8,16,32,40,48,64,72,80,96,112,128,144,160,192,240,256,264,272,288,304,320,352,384,448,480,544,576,960,1920,15360}

set(5)={-860160,-53760,-23040,-19680,-18728,-17688,-17104,-16744,-16400,-16256,-15968,-15808,-15632,-14808,-14640,-14624,-14048,-13824,-13760,-13608,-13184,-12160,-11520,-11264,-10720,-10448,-10368,-10240,-10088,-10036,-9600,-9248,-8960,-8832,-8640,-8448,-8416,-7760,-6400,-6144,-5760,-5728,-5600,-5568,-5536,-5504,-5376,-5280,-5248,-5184,-5104,-5024,-4984,-4976,-4928,-4864,-4832,-4800,-4704,-4608,-4568,-4544,-4472,-4352,-4320,-4272,-4224,-4160,-4128,-4064,-3968,-3936,-3872,-3808,-3792,-3776,-3680,-3648,-3632,-3584,-3568,-3552,-3520,-3456,-3408,-3328,-3264,-3224,-3072,-3008,-2928,-2920,-2912,-2896,-2888,-2880,-2784,-2776,-2752,-2712,-2688,-2624,-2560,-2528,-2504,-2464,-2432,-2416,-2408,-2400,-2392,-2368,-2336,-2320,-2304,-2288,-2264,-2240,-2224,-2208,-2176,-2144,-2112,-2096,-2080,-2072,-2064,-2048,-2024,-2016,-1992,-1984,-1920,-1904,-1888,-1872,-1856,-1840,-1824,-1808,-1792,-1760,-1728,-1712,-1696,-1680,-1672,-1664,-1648,-1632,-1624,-1616,-1600,-1584,-1568,-1536,-1508,-1504,-1488,-1472,-1440,-1432,-1424,-1408,-1392,-1376,-1344,-1328,-1312,-1304,-1296,-1288,-1280,-1272,-1264,-1256,-1248,-1232,-1224,-1216,-1200,-1184,-1176,-1168,-1152,-1120,-1112,-1104,-1088,-1080,-1072,-1056,-1048,-1040,-1032,-1024,-1016,-1008,-992,-984,-976,-960,-952,-936,-928,-912,-896,-880,-864,-848,-832,-816,-808,-800,-788,-784,-768,-760,-752,-744,-736,-728,-720,-712,-704,-696,-688,-672,-664,-656,-648,-640,-632,-624,-616,-608,-600,-592,-584,-576,-560,-552,-544,-540,-536,-528,-524,-516,-512,-504,-496,-492,-480,-476,-472,-468,-464,-456,-448,-440,-432,-424,-416,-408,-400,-392,-384,-376,-368,-360,-352,-344,-336,-328,-320,-312,-304,-296,-288,-280,-272,-264,-256,-248,-240,-232,-224,-216,-208,-200,-192,-184,-176,-168,-164,-160,-152,-148,-144,-136,-132,-128,-120,-116,-112,-104,-100,-96,-92,-88,-84,-80,-76,-72,-68,-64,-60,-56,-52,-48,-44,-40,-36,-32,-28,-24,-20,-16,-12,-8,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,112,116,120,128,132,136,144,148,152,160,164,168,176,184,192,200,208,216,224,232,240,248,256,264,272,280,288,296,304,312,320,328,336,344,352,360,368,376,384,392,400,408,416,424,432,440,448,456,464,468,472,476,480,492,496,504,512,516,524,528,536,540,544,552,560,576,584,592,600,608,616,624,632,640,648,656,664,672,688,696,704,712,720,728,736,744,752,760,768,776,784,800,816,832,848,864,880,896,912,928,936,952,960,976,984,992,1008,1016,1024,1032,1040,1048,1056,1072,1080,1088,1104,1112,1120,1152,1168,1176,1184,1200,1216,1232,1248,1264,1272,1280,1288,1304,1312,1328,1344,1376,1392,1408,1424,1440,1472,1488,1504,1508,1536,1552,1568,1576,1584,1600,1616,1640,1648,1664,1680,1696,1712,1728,1760,1792,1800,1808,1856,1872,1888,1904,1920,1968,1984,1992,2016,2024,2048,2064,2072,2080,2096,2112,2116,2144,2176,2208,2224,2240,2264,2272,2288,2304,2320,2344,2352,2368,2400,2408,2432,2496,2504,2528,2560,2592,2608,2688,2704,2712,2752,2768,2776,2784,2880,2928,2968,3072,3096,3104,3168,3200,3224,3264,3392,3456,3520,3552,3568,3584,3632,3648,3664,3712,3744,3760,3776,3808,3840,3872,3936,4064,4096,4160,4224,4272,4320,4352,4416,4472,4480,4544,4568,4576,4608,4656,4688,4704,4800,4840,5024,5104,5120,5152,5184,5248,5280,5376,5440,5536,5568,5600,5728,5760,5952,6016,6288,6528,7680,7760,8416,8640,8832,9216,9248,9600,11520,13608,14048,15632,15808,16256,23040,46080,69120,70080,70272,70880,71280}

Key directions to explore:
1. The integer with the highest absolute value at any L is given by (-4)**L/2*math.factorial(2*(L-1))/math.factorial(L-1)
2. The highest integer satisfies a simple recursion relation
3. Try recursive approaches for the whole sets.
4. The numbers often, but not always, have simple prime number decomposition.
5. The combinatorics might be related to the restricted Dyck Path.

ALWAYS make the patchname a string.
Be creative and try to find a new solution better than the best known result."""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=400,
    max_parallel_jobs=5,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "o4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto", "low", "medium", "high"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gpt-5-nano"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gpt-5-nano"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path="initial.py",
    results_dir="results_coefficients",
)


def main():
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    results_data = main()
