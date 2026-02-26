# some different lossfunction (make own class in .py file to call in model .py file)
import tensorflow as tf

def loss_function(y_true, y_pred):
    label_y_true = y_true[y_true == 1]
    label_y_pred = y_pred[y_true == 1]
    background_y_true = y_true[y_true == 0]
    background_y_pred = y_pred[y_true == 0]

    label_loss = tf.keras.losses.MSE(label_y_true, label_y_pred)
    background_loss = tf.keras.losses.MSE(background_y_true, background_y_pred)
    # as both losses are mean values, no weights are needed. Bc summing them makes them equally valuable
    loss = label_loss + background_loss
    return loss


def weighted_bce_loss(pos_weight):
    def loss(y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight))

    return loss


def adaptive_wing_loss(y_true, y_pred, omega=14, theta=0.5, epsilon=1, alpha=2.1):
    # omega, theta, epsilon, alpha are hyperparameters
    # that control the curvature of the loss.

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    delta_y = tf.abs(y_true - y_pred)

    # Adaptive factor
    A = omega * (1.0 / (1.0 + tf.pow(theta / epsilon, alpha - y_true))) * (alpha - y_true) * tf.pow(theta / epsilon,
                                                                                                    alpha - y_true - 1.0) * (
                    1.0 / epsilon)
    C = (theta * A - omega * tf.math.log(1.0 + tf.pow(theta / epsilon, alpha - y_true)))

    loss = tf.where(delta_y < theta,
                    omega * tf.math.log(1.0 + tf.pow(delta_y / epsilon, alpha - y_true)),
                    A * delta_y - C)

    return tf.reduce_mean(loss)
