import xml.dom.minidom

def gpu_config(gpu_num):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def get_test_data(save_file):
    domtree = xml.dom.minidom.parse(save_file)
    seq_writer = open('../decoder/seq_crt_q.txt','w')
    idx_writer = open('../decoder/idx_crt_q.txt','w')
    segs = domtree.documentElement.getElementsByTagName('seg')
    for i, seg in enumerate(segs):
        DocID, SenID, EngSen = seg.childNodes[0].data.split('\t')
        seq_writer.write(EngSen.strip()+'\n')
        idx_writer.write(str(i+1)+'\t'+str(DocID)+'\t'+str(SenID)+'\n')
    seq_writer.close()
    idx_writer.close()

if __name__ =='__main__':
    get_test_data( "../rawdata/test_data/crt_en.sgm" )
