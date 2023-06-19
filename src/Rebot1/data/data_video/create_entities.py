from create_keywords_modified import extract_keywords, save_clip_embeddings, save_image_embeddings, dict_to_tensor, select_topk, create_clip_Da_dataset_from_saved, create_titles, create_clip_Da_dataset_from_saved_nlvr
import sys

data_index = sys.argv[1]

entities_path = '/mmu_nlp/wuxing/yuhuimu/ViCHA/preprocess/unigrams.txt'
kwords_path = '/mmu_nlp/wuxing/yuhuimu/ViCHA/dataset/augmented_data/keywords_t3.json'
kwords_embed_path = '/mmu_nlp/wuxing/yuhuimu/ViCHA/dataset/augmented_data/keywords_t3_embeddings.json'
csv_path = '/home/wuxing/datasets/WebVid/metadata/results_2M_train.csv'
video_embed_path = '/mmu_nlp/wuxing/yuhuimu/ViCHA/dataset/augmented_data/train_video_embeddings.json'
video_root = '/home/wuxing/datasets/WebVid/videos'
output_path = '/mmu_nlp/wuxing/yuhuimu/ViCHA/dataset/augmented_data/filter_train.csv'

# objs, atts, rels = extract_keywords(entities_path, extract_rel=False, extract_att=False, output_path=kwords_path)

# text_embed = save_clip_embeddings(kwords_path, kwords_embed_path)

tmp = save_image_embeddings(csv_path, output_path=video_embed_path, video_root=video_root, overwrite=True, snli=False)

tmp = create_clip_Da_dataset_from_saved(csv_path, kwords_embed_path, video_embed_path, k=15, output_path=output_path, image_root=None, overwrite=True)

