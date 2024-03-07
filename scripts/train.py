import diffuser.utils as utils
import pdb


import random
random.seed(42)    ### TQ Attempt to stabalise the outputs

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1' #  'maze2d-medium-v1' 'maze2d-umaze-v1'  'maze2d-large-v1'   
    config: str = 'config.maze2d'         
    
    #dataset: str = 'breakout-mixed-v0'  # Example dataset name, adjust based on actual availability
    #config: str = 'config.atari.breakout'  # Example config path, adjust based on your setup
 
    #dataset: str = 'maze2d-umaze-v1'        
    #config: str = 'config.maze2d'  #      minigrid-fourrooms-v0
 
    ### It trains on the Locomtion data --    
    #dataset: str = 'hopper-medium-expert-v2'        ### It trains on the Locomtion data -- 
    #config: str = 'config.locomotion'
    
args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
print("args.loader:",args.loader)
print("env:", args.dataset)
print("args.horizon:", args.horizon)
print("args.preprocess_fns:", args.preprocess_fns)
 
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,              
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)
 
dataset = dataset_config()          ## Call the DATASET CONFIG Function GET and Configure the data in a certain way - Grab the Data
renderer = render_config()
 
print("*datasetAAA",dataset)
#print("*dataset[1]",dataset[1])
  
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
 
#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

print("*datasetAA",dataset)
 
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

print("###### The Model ######")

print("*datasetA",dataset)

model = model_config()
 
print("###### The diffusion_config ######")
 
print("*datasetB",dataset)
 
diffusion = diffusion_config(model)
 
print("*dataset[0]",dataset[0])   ### Changed to an output of 6 here, actions then positions then velocities   
                                  ## Hopper is 14 - 3 actions + 11 observateions - must be maze2D by default 
 
print("###### The trainer_config ######")
 
trainer = trainer_config(diffusion, dataset, renderer)      ## DataSet INSERTED INTO THE TRAINING 

#-----------------------------------------------------------------------------#
#------------------------ Maybe Just a *Test* Set forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#
  
utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)

### AN IDEA - JUST OVERWRITE THE DATASET ONE BATCH AT A TIME HERE?! OVERWRTIE THE VELOCITIES AND ROTATE THE POSITIONS!? ### 
 
#print("orig Dataset", dataset )

#print("*dataset[0]",dataset[0])
 
batch = utils.batchify(dataset[0])
 
#print("orig Dataset", dataset )

#print("batch = ", batch) 
#print("batch = ", batch.size)   ## Batch is a TENSOR
       


#batch = dataset[1]
#trajectories = batch.trajectories
 
# Get the number of samples and features
#num_samples, num_features = trajectories.shape
  
#print(f"Number of samples: {num_samples}")
#print(f"Number of features: {num_features}")
 
### Need to try elininating a column for ALL datasets!! 
 
#dataset[0][1][1] = 0.1123
print("*dataset[0]",dataset[0]) 
 
#nprint("*dataset[0]",dataset[0]['Observations']) 

for i in range(2):
    print(i)  
    print("*dataset[i]",dataset[i])



print(type(dataset[0]))
 
#print("*dataset[0].shape",dataset[0].size) 
#print("*dataset[1].shape",dataset[1].size) 
 
#if hasattr(dataset[0], 'trajectories'):
#    trajectories = dataset[0].trajectories()
#    print(trajectories)  
   
print("* NEW dataset[0]",dataset[0])
 
#print("*dataset[0]",dataset[0].shape)
#print("*dataset[0]",dataset[1].shape)
print("*dataset overall ",dataset) 
#print("*dataset",dataset.shape)

##### ACTUALLY RUNS Diffusion  ##### 
                                             #dataset[trajectories[0]["observations"]]
loss, _ = diffusion.loss(*batch)     ###  Calls LOSS and PLots for this batch
print("*batch",*batch)      
print("*loss = ",loss)      
loss.backward()                       ### Pefroms Gradient Descent!
print('✓ YES TEST PASSED!')

 
#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

#Finalized replay buffer | 1566 episodes

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)   ### 200 epochs, 10,000 steps per epoch

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)      ### Runs the training - END OF THE CODE!!!! 

