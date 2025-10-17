while read requirement; do
    # 去掉注释和空行
    req=$(echo $requirement | sed 's/#.*//g' | xargs)
    if [ -n "$req" ]; then
        pip show $req > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Installing $req ..."
            pip install "$req"
        else
            echo "Skipping $req (already installed)"
        fi
    fi
done < requirements.txt