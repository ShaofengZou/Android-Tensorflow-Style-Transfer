package com.shaofengzou.styletransfer;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.Toast;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import java.util.List;

public class StyleAdapter extends RecyclerView.Adapter<StyleAdapter.ViewHolder>{

    private List<Style> mStyleList;

    class ViewHolder extends RecyclerView.ViewHolder {
        View styleView;
        ImageView styleImage;

        public ViewHolder(View view) {
            super(view);
            styleView = view;
            styleImage = (ImageView) view.findViewById(R.id.style_image);
        }

    }

    public StyleAdapter(List<Style> fruitList) {
        mStyleList = fruitList;
    }

    //define interface
    public static interface OnItemClickListener {
        void onItemClick(View view , int position);
    }

    private OnItemClickListener mOnItemClickListener = null;

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.style_item, parent, false);
        final ViewHolder holder = new ViewHolder(view);

        holder.styleImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int position = holder.getAdapterPosition();
                Style style = mStyleList.get(position);

                if (mOnItemClickListener != null) {
                    //注意这里使用getTag方法获取position
                    mOnItemClickListener.onItemClick(v,position);
                }




            }
        });
        return holder;
    }

    @Override
    public void onBindViewHolder(ViewHolder holder, int position) {
        Style style = mStyleList.get(position);
        holder.styleImage.setImageResource(style.getImageId());

        //将position保存在itemView的Tag中，以便点击时进行获取
        holder.itemView.setTag(position);
    }

    @Override
    public int getItemCount() {
        return mStyleList.size();
    }

    public void setOnItemClickListener(OnItemClickListener listener) {
        this.mOnItemClickListener = listener;
    }
}