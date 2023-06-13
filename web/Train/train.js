const saveModelData = () => {
  const modelConfig = {
    model_to_train: document.getElementById('model_to_train').value,
    masking_image: document.getElementById('masking_image').value,
    edge_detector: document.getElementById('edge_detector').value,
    train_list: document.getElementById('train_list').value,
    value_list: document.getElementById('value_list').value,
    test_list: document.getElementById('test_list').value ,
  }
  eel.save_model_config(modelConfig);
};

const saveGANData = () => {
  const modelConfig = {
    lr: document.getElementById('lr').value,
    g_d_lr: document.getElementById('g_d_lr').value,
    adam_op: document.getElementById('adam_op').value,
    b_size: document.getElementById('b_size').value,
    image_size: document.getElementById('image_size').value,
    sigma: document.getElementById('sigma').value,
    gan_l: document.getElementById('gan_l').value,
    max_iterations: document.getElementById('max_iterations').value,
    edge_threshold: document.getElementById('edge_threshold').value,
    l1_loss_w: document.getElementById('l1_loss_w').value,
    fm_loss_w: document.getElementById('fm_loss_w').value,
    content_loss_w: document.getElementById('content_loss_w').value,
    style_loss_w: document.getElementById('style_loss_w').value,
    inpaint_adv_loss_w: document.getElementById('inpaint_adv_loss_w').value,
  }
  eel.save_model_config(modelConfig);
};

