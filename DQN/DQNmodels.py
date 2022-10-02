import tensorflow as tf
# import tensorflow.contrib.layers as layers
import tensorflow.compat.v1.keras.layers as layers


def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from Common.CommonModels import get_network_builder
        print("our network kkkkkkkkkkkkkkkk",network)
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            latent = network(input_placeholder)
            print("latent",latent)
            if isinstance(latent, tuple):
                if latent[1] is not None:
                    raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                latent = latent[0]

            # latent = layers.flatten(latent)
            latent = layers.Flatten()(latent)
            print("here",latent)

            with tf.compat.v1.variable_scope("action_value"):
                action_out = latent
                for hidden in hiddens:
                    # action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    #tf.compat.v1.layers.dense()
                    """
                    tf.layers.dense(
                        inputs, units, activation=None, use_bias=True, kernel_initializer=None,
                        bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                        bias_constraint=None, trainable=True, name=None, reuse=None
                    )
                    
                    """
                    action_out = tf.compat.v1.layers.dense(action_out, hidden, activation=None)
                    if layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                # action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)
                action_scores = tf.compat.v1.layers.dense(action_out,num_actions, activation=None)

            if dueling:
                with tf.compat.v1.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        # state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                        state_out = tf.compat.v1.layers.dense(state_out, hidden, activation=None)
                        if layer_norm:
                            state_out = layers.layer_norm(state_out, center=True, scale=True)
                        state_out = tf.nn.relu(state_out)
                    # state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                    state_score = tf.compat.v1.layers.dense(state_out, 1, activation=None)
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores
            return q_out

    return q_func_builder
